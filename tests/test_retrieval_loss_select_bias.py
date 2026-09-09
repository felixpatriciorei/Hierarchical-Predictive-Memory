"""Regression test for the retrieval_loss / select_bias interaction bug.

Found 2026-08-02 after a control run (learned_writer_teacher_forcing pinned
on for the whole run) recovered eval_exact to ~0.6, isolating the collapse
to the differentiable full-candidate read path. Root cause: EpisodicMemory
.forward computed retrieval_loss as F.cross_entropy(scores[valid],
positive[valid]) where `scores` already has `select_bias` added (via
retrieve_topk). select_bias subtracts 1e4 from every valid-but-unselected
candidate. Whenever the writer hasn't selected the oracle's true-fact
position -- common early in training, see writer_missed_fact_rate in the
raw sweep logs -- the target class's own logit is suppressed by 1e4 before
cross_entropy ever sees it, producing a loss spike of roughly that
magnitude with no relationship to how well query/key matching actually
works. This is what produced the "loss": 300-1400 explosions the instant
learned_writer_teacher_forcing turned off, identically for jepa_bias_on
and jepa_bias_off (JEPA was never involved).

Fix: recompute retrieval_loss on raw, select_bias-free content-matching
scores. select_bias still shapes the actual retrieved output/weights via
retrieve_topk -- only the loss target changed.
"""
import torch
import torch.nn.functional as F

from hpm.memory import EpisodicMemory


def test_retrieval_loss_is_not_spiked_by_an_unselected_true_fact():
    torch.manual_seed(0)
    bsz, num_candidates, d_model = 2, 6, 8
    mem = EpisodicMemory(d_model=d_model)

    # 8 embeddable positions per batch row: enough for memory_token_positions
    # (num_candidates pairs of [key_pos, value_pos]) plus a query position.
    token_embeddings = torch.randn(bsz, num_candidates + 2, d_model)
    starts = torch.arange(num_candidates)
    memory_token_positions = torch.stack([starts, starts + 0], dim=-1)[None].expand(bsz, num_candidates, 2).contiguous()
    memory_mask = torch.ones(bsz, num_candidates, dtype=torch.bool)
    query_key_positions = torch.full((bsz,), num_candidates, dtype=torch.long)

    # Oracle says candidate index 2 is the true fact for every row.
    positive = torch.full((bsz, 1), 2, dtype=torch.long)

    # select_bias mimics the writer having picked candidates {0, 1} instead
    # of the true fact at index 2: 0 bias for selected, -1e4 for the rest
    # (exactly differentiable_topk_bias's hard-component convention).
    select_bias = torch.full((bsz, num_candidates), -1.0e4)
    select_bias[:, 0] = 0.0
    select_bias[:, 1] = 0.0

    _, info_biased = mem(
        token_embeddings, memory_token_positions, memory_mask, query_key_positions,
        top_k=1, num_hops=1, hop_positive_indices=positive, select_bias=select_bias,
    )
    _, info_unbiased = mem(
        token_embeddings, memory_token_positions, memory_mask, query_key_positions,
        top_k=1, num_hops=1, hop_positive_indices=positive, select_bias=None,
    )

    # The whole point of the fix: retrieval_loss must not care whether the
    # true fact was among the writer's selected candidates. With and
    # without select_bias, the loss should be close (same underlying
    # content-matching scores) -- NOT off by ~1e4, which is what the
    # pre-fix code produced (cross_entropy against a target logit
    # suppressed by exactly that amount).
    assert torch.isfinite(info_biased["retrieval_loss"])
    diff = (info_biased["retrieval_loss"] - info_unbiased["retrieval_loss"]).abs().item()
    assert diff < 10.0, (
        f"retrieval_loss changed by {diff} when select_bias excluded the true fact -- "
        "this is the 1e4-magnitude spike bug; retrieval_loss must be computed on "
        "select_bias-free content-matching scores"
    )
    assert info_biased["retrieval_loss"].item() < 50.0, (
        "retrieval_loss should stay near normal cross-entropy magnitude "
        "(ln(num_candidates)-ish), not spike toward ~1e4"
    )


def test_writer_still_gets_gradient_through_retrieval_loss():
    """The fix must not repeat the mistake of a naive raw-scores-only patch,
    which silenced retrieval_loss's gradient into select_bias entirely (a
    real regression -- caught by
    test_differentiable_writer_selection.test_writer_gradient_from_real_training_loss,
    since at top_k=1 answer_loss alone is exactly zero-gradient into the
    writer -- softmax-of-one -- making retrieval_loss the only remaining
    channel). Confirms retrieval_loss's gradient still reaches a select_bias
    tensor that requires grad, even though its forward VALUE (tested above)
    no longer spikes.
    """
    torch.manual_seed(1)
    bsz, num_candidates, d_model = 2, 6, 8
    mem = EpisodicMemory(d_model=d_model)

    token_embeddings = torch.randn(bsz, num_candidates + 2, d_model)
    starts = torch.arange(num_candidates)
    memory_token_positions = torch.stack([starts, starts + 0], dim=-1)[None].expand(bsz, num_candidates, 2).contiguous()
    memory_mask = torch.ones(bsz, num_candidates, dtype=torch.bool)
    query_key_positions = torch.full((bsz,), num_candidates, dtype=torch.long)
    positive = torch.full((bsz, 1), 2, dtype=torch.long)

    select_bias = torch.zeros(bsz, num_candidates, requires_grad=True)

    _, info = mem(
        token_embeddings, memory_token_positions, memory_mask, query_key_positions,
        top_k=1, num_hops=1, hop_positive_indices=positive, select_bias=select_bias,
    )
    info["retrieval_loss"].backward()
    assert select_bias.grad is not None and select_bias.grad.abs().sum() > 0, (
        "retrieval_loss produced zero gradient into select_bias -- this is the "
        "regression a naive raw-scores-only fix causes; the writer must still "
        "learn from the real downstream loss, not just its own oracle BCE"
    )
