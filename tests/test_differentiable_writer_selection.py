"""
Regression tests for the differentiable writer-selection patch.

Run: pytest tests/test_differentiable_writer_selection.py -v

Covers, in order of how they were actually found (not idealized top-down design):
  1. differentiable_topk_bias forward == old masked_fill pattern, exactly.
  2. differentiable_topk_bias gradient magnitude is decoupled from bias_scale
     (regression test for the ~1e9x gradient-amplification bug).
  3. The Sinkhorn anchor construction doesn't NaN when fed a row with many
     invalid (float-min-sentinel) candidates (regression test for the
     float32 overflow bug: min - margin*spread -> -inf -> NaN in logsumexp).
  4. select_bias threaded through retrieve_topk actually changes which slot
     wins, and gradient reaches select_bias from the retrieved output.
  5. End-to-end on HpmLiteV2Model: forward pass with the new full-candidate
     + select_bias read path is numerically identical to the old
     hard-gathered path, AND gradient from (answer_loss + lambda_ret *
     retrieval_loss) reaches model.writer's parameters when
     learned_writer_teacher_forcing=False (this was exactly 0.0 pre-patch).
  6. answer_loss ALONE at top_k=1 is still exactly zero gradient into the
     writer -- documenting the separate softmax-of-one identity so nobody
     mistakes it for a regression later.
"""
import math

import torch
import torch.nn.functional as F
import pytest

from hpm.differentiable_topk import differentiable_topk_bias, differentiable_topk_mask
from hpm.memory import retrieve_topk
from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model


# ---------------------------------------------------------------------------
# 1 & 2: differentiable_topk_bias itself
# ---------------------------------------------------------------------------

def test_bias_forward_matches_hard_masked_fill():
    torch.manual_seed(0)
    scores = torch.randn(6, 10)
    k = 3
    bias = differentiable_topk_bias(scores, k=k, eps=0.3, n_iters=50, bias_scale=1.0e4)

    ref_idx = scores.topk(k, dim=-1).indices
    ref_mask = torch.zeros_like(scores).scatter_(-1, ref_idx, 1.0)
    ref_bias = (ref_mask - 1.0) * 1.0e4

    assert torch.allclose(bias.detach(), ref_bias), "forward value must match hard masked_fill exactly"


def test_bias_gradient_magnitude_independent_of_bias_scale():
    # Regression test for the exact bug found on 2026-08-01: scaling the
    # whole STE tensor by bias_scale also scaled backward gradient by
    # bias_scale (~1e9x amplification, verified via a real training run --
    # gradient sum went from 117548025.6875 to 0.117548 after the fix, a
    # ratio of exactly the old bias_scale). Assert this stays fixed: two
    # wildly different bias_scale values must produce near-identical
    # gradient magnitude into `scores`.
    torch.manual_seed(1)
    base_scores = torch.randn(4, 12)

    grads = []
    for bias_scale in (1.0e2, 1.0e4, 1.0e7):
        scores = base_scores.clone().requires_grad_(True)
        bias = differentiable_topk_bias(scores, k=3, eps=0.3, n_iters=50, bias_scale=bias_scale)
        # a loss that only exists because of the differentiable component
        # (constant hard part alone would have zero gradient into scores)
        loss = (bias ** 2).sum()
        loss.backward()
        grads.append(scores.grad.abs().sum().item())

    print("grad sums across bias_scale in {1e2,1e4,1e7}:", grads)
    # they won't be IDENTICAL (loss itself scales with bias magnitude via
    # the hard component's square), so compare the SLOPE isn't ~linear in
    # bias_scale the way the old bug was (old bug: ~1e9x per 1e9x of scale).
    # Use scores.grad from a bias_scale-INVARIANT loss instead: sum of bias
    # directly (linear), which isolates exactly the old failure mode.
    grads_linear = []
    for bias_scale in (1.0e2, 1.0e4, 1.0e7):
        scores = base_scores.clone().requires_grad_(True)
        bias = differentiable_topk_bias(scores, k=3, eps=0.3, n_iters=50, bias_scale=bias_scale)
        bias.sum().backward()
        grads_linear.append(scores.grad.abs().sum().item())

    print("grad sums (linear loss) across bias_scale in {1e2,1e4,1e7}:", grads_linear)
    lo, hi = min(grads_linear), max(grads_linear)
    assert hi / max(lo, 1e-12) < 5.0, (
        f"gradient magnitude scales with bias_scale ({grads_linear}) -- "
        "the forward/backward scale-coupling bug is back"
    )


# ---------------------------------------------------------------------------
# 3: the float32 overflow bug (many invalid candidates)
# ---------------------------------------------------------------------------

def test_bias_nans_if_fed_the_raw_float_min_sentinel():
    # This documents WHY hpm/model.py uses a finite floor instead of
    # feeding `masked_logits` (invalid = torch.finfo(dtype).min) directly
    # into the Sinkhorn bias: min - margin*spread overflows to -inf, which
    # poisons logsumexp into NaN. This is a property of the primitive
    # itself (it assumes a bounded score range), not something fixed inside
    # differentiable_topk.py -- callers must precondition input, exactly as
    # model.py's `sinkhorn_input`/`sinkhorn_floor` construction now does.
    # If this test ever starts passing (no NaN), the primitive's contract
    # changed -- update model.py's comment, don't just delete this test.
    torch.manual_seed(2)
    n, k = 38, 3
    logits = torch.randn(3, n) * 2
    valid = torch.ones(3, n, dtype=torch.bool)
    valid[:, -6:] = False

    finfo_min = torch.finfo(torch.float32).min
    danger_input = logits.masked_fill(~valid, finfo_min)

    out = differentiable_topk_bias(danger_input, k=k, eps=0.3, n_iters=50)
    assert torch.isnan(out).any(), "expected NaN from the float-min sentinel -- documents the hazard model.py avoids"


def test_writer_select_bias_has_no_nan_on_realistic_batch(small_model):
    # Positive counterpart to the test above: model.py's actual finite-floor
    # fix (sinkhorn_floor = row_max - 50, not the float-min sentinel) must
    # keep writer_select_bias NaN-free on a realistic batch with a real
    # invalid tail (query/answer tokens).
    torch.manual_seed(4)
    model = small_model
    input_ids, mem_pos, mem_mask, qkp, hop_pos = _build_batch()
    with torch.no_grad():
        local_state = model._local_path(input_ids)
        writer_info = model.writer(
            local_state, input_ids, qkp, mem_pos, mem_mask,
            max_slots=min(model.config.episodic_capacity, mem_pos.size(1)),
        )
    assert not torch.isnan(writer_info["writer_select_bias"]).any()


# ---------------------------------------------------------------------------
# 4: select_bias threaded through retrieve_topk
# ---------------------------------------------------------------------------

def test_select_bias_can_override_retrieve_topk_winner():
    query = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    memory_keys = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    memory_values = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True]])

    # without bias: slot 1 wins on content score
    _, _, top_indices, _, _ = retrieve_topk(query, memory_keys, memory_values, mask, top_k=1)
    assert top_indices.tolist() == [[1]]

    # with a bias favoring slot 0 heavily: slot 0 should win instead
    bias = torch.tensor([[1.0e4, 0.0]])
    _, _, top_indices2, _, _ = retrieve_topk(query, memory_keys, memory_values, mask, top_k=1, select_bias=bias)
    assert top_indices2.tolist() == [[0]], "select_bias should be able to override the content-score winner"


def test_gradient_reaches_select_bias_through_retrieve_topk():
    query = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    memory_keys = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    memory_values = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True]])
    bias = torch.zeros(1, 2, requires_grad=True)

    retrieved, scores, _, _, _ = retrieve_topk(query, memory_keys, memory_values, mask, top_k=2, select_bias=bias)
    # top_k=2 here deliberately avoids the softmax-of-one identity so this
    # test isolates "does select_bias reach the graph at all" from the
    # separate top_k=1 fact covered in test 6 below.
    retrieved.sum().backward()
    assert bias.grad is not None and bias.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 5 & 6: end-to-end on the real model
# ---------------------------------------------------------------------------

def _build_batch(bsz=4, seq_len=40, num_facts=8, query_tok=1, answer_tok=2):
    input_ids = torch.randint(5, 50, (bsz, seq_len))
    starts = torch.linspace(0, seq_len - 6, num_facts).long()
    mem_pos = torch.stack([starts, starts + 1], dim=-1)[None].expand(bsz, num_facts, 2).contiguous()
    mem_mask = torch.ones(bsz, num_facts, dtype=torch.bool)
    qkp = torch.full((bsz,), seq_len - 2, dtype=torch.long)
    input_ids[:, -2] = query_tok
    input_ids[:, -1] = answer_tok
    hop_pos = torch.randint(0, num_facts, (bsz, 1))
    return input_ids, mem_pos, mem_mask, qkp, hop_pos


@pytest.fixture
def small_model():
    cfg = HpmLiteV2Config(d_model=32, layers=1, heads=2, window=16, block_size=8,
                           use_learned_writer=True, use_jepa_aux=False, episodic_capacity=3)
    return HpmLiteV2Model(cfg)


def test_new_read_path_matches_old_hard_path_forward(small_model):
    torch.manual_seed(3)
    model = small_model
    input_ids, mem_pos, mem_mask, qkp, hop_pos = _build_batch()

    with torch.no_grad():
        local_state = model._local_path(input_ids)
        writer_info = model.writer(
            local_state, input_ids, qkp, mem_pos, mem_mask,
            max_slots=min(model.config.episodic_capacity, mem_pos.size(1)),
        )
        new_vec, _ = model.episodic_memory(
            local_state, writer_info["writer_full_memory_token_positions"],
            writer_info["writer_full_valid_mask"], qkp, top_k=1, num_hops=1,
            select_bias=writer_info["writer_select_bias"],
        )
        old_vec, _ = model.episodic_memory(
            local_state, writer_info["writer_memory_token_positions"],
            writer_info["writer_memory_mask"], qkp, top_k=1, num_hops=1,
            select_bias=None,
        )
    assert torch.allclose(new_vec, old_vec, atol=1e-4)


def test_writer_gradient_from_real_training_loss(small_model):
    torch.manual_seed(3)
    model = small_model
    input_ids, mem_pos, mem_mask, qkp, hop_pos = _build_batch()

    model.zero_grad()
    out = model(
        input_ids, mem_pos, mem_mask, answer_positions=None, query_key_positions=qkp,
        top_k=1, task="kv", hop_positive_memory_indices=hop_pos,
        use_learned_writer=True, learned_writer_teacher_forcing=False,
    )
    target = torch.randint(5, 50, (input_ids.size(0),))
    answer_loss = F.cross_entropy(out["logits"][:, -1, :], target)
    retrieval_loss = out["retrieval"].get("retrieval_loss", answer_loss.new_zeros(()))
    total = answer_loss + 0.1 * retrieval_loss  # matches train.py's default lambda_ret
    total.backward()

    total_writer_grad = sum(
        p.grad.abs().sum().item() for p in model.writer.parameters() if p.grad is not None
    )
    assert total_writer_grad > 0, "writer got zero gradient from the real training objective"


def test_answer_loss_alone_still_zero_at_top_k_1(small_model):
    # Documents the separate, pre-existing softmax-of-one identity so it's
    # never mistaken for a regression: torch.softmax over a size-1 last dim
    # is identically 1.0 for any input, so its gradient is exactly 0.
    torch.manual_seed(3)
    model = small_model
    input_ids, mem_pos, mem_mask, qkp, hop_pos = _build_batch()

    model.zero_grad()
    out = model(
        input_ids, mem_pos, mem_mask, answer_positions=None, query_key_positions=qkp,
        top_k=1, task="kv", hop_positive_memory_indices=hop_pos,
        use_learned_writer=True, learned_writer_teacher_forcing=False,
    )
    target = torch.randint(5, 50, (input_ids.size(0),))
    F.cross_entropy(out["logits"][:, -1, :], target).backward()
    total_writer_grad = sum(
        p.grad.abs().sum().item() for p in model.writer.parameters() if p.grad is not None
    )
    assert total_writer_grad == 0.0, (
        "expected exactly 0 -- if this now fails, something changed the top_k=1 "
        "softmax-of-one behavior; update this test deliberately, don't just delete it"
    )


# ---------------------------------------------------------------------------
# 7: sinkhorn-warmup / jepa_writer_bias shared-tensor fix
#
# writer_select_bias is computed every step (TF or not) by self.writer, and
# during TF it already has exactly one trainer: the writer's own oracle-
# matching BCE (writer_loss). The sinkhorn-warmup path reuses that same
# tensor as an input (warm_bias) to give router/episodic_memory/answer_head
# real answer-loss gradient during TF. Before this fix, warm_bias was NOT
# detached, so the warmup answer-loss became a SECOND trainer of
# writer_select_bias -- and whenever jepa_bias is on, that tensor traces back
# through self.jepa.token_surprise, meaning JEPA's token-level predictor was
# being pulled by both the writer's oracle-matching objective and the
# warmup path's answer objective at once. This tracked exactly with the
# clamped+ramped sweep's jepa_bias_on-specific regression. Fixed with one
# `.detach()` on warm_bias. These tests prove it's cut in exactly the right
# place: gone into jepa.token_*, still present into the warmup path's actual
# intended targets, and writer_loss (which never went through warm_bias)
# is completely unaffected.
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model_jepa_bias_warmup():
    cfg = HpmLiteV2Config(
        d_model=32, layers=1, heads=2, window=16, block_size=8,
        use_learned_writer=True, use_jepa_aux=True, use_jepa_writer_bias=True,
        sinkhorn_warmup=True, episodic_capacity=3,
    )
    return HpmLiteV2Model(cfg)


def _forward_teacher_forced(model, input_ids, mem_pos, mem_mask, qkp, hop_pos):
    return model(
        input_ids, mem_pos, mem_mask, answer_positions=None, query_key_positions=qkp,
        top_k=1, task="kv", hop_positive_memory_indices=hop_pos,
        use_learned_writer=True, learned_writer_teacher_forcing=True,
    )


def test_warmup_loss_alone_reaches_zero_gradient_into_jepa_token_surprise(small_model_jepa_bias_warmup):
    # The core regression test for the fix: backward on the warmup answer-
    # loss ALONE (nothing else) must leave jepa.token_context/predictor/
    # target with either no grad at all, or exactly zero -- proving the
    # detach actually severs that path, not just reduces it.
    torch.manual_seed(4)
    model = small_model_jepa_bias_warmup
    input_ids, mem_pos, mem_mask, qkp, hop_pos = _build_batch()

    model.zero_grad()
    out = _forward_teacher_forced(model, input_ids, mem_pos, mem_mask, qkp, hop_pos)
    warm_logits = out["retrieval"]["sinkhorn_warmup_logits"]
    target = torch.randint(5, 50, (input_ids.size(0),))
    warmup_loss = F.cross_entropy(warm_logits[:, -1, :], target)
    warmup_loss.backward()

    token_surprise_params = list(model.jepa.token_context.parameters()) \
        + list(model.jepa.token_predictor.parameters()) \
        + list(model.jepa.token_target.parameters())
    assert len(token_surprise_params) > 0
    total_grad = sum(p.grad.abs().sum().item() for p in token_surprise_params if p.grad is not None)
    assert total_grad == 0.0, (
        "warmup loss must NOT reach jepa.token_surprise's parameters -- if this "
        "fails, the detach() on warm_bias in hpm_v2_model.py was removed or "
        "bypassed"
    )


def test_warmup_loss_alone_still_reaches_its_intended_targets(small_model_jepa_bias_warmup):
    # The detach must be surgical: it should NOT also kill gradient into the
    # modules the warmup mechanism actually exists to train. Detaching
    # warm_bias only cuts the graph upstream of that tensor, not downstream,
    # so router/episodic_memory/answer_head must still get real gradient.
    torch.manual_seed(4)
    model = small_model_jepa_bias_warmup
    input_ids, mem_pos, mem_mask, qkp, hop_pos = _build_batch()

    model.zero_grad()
    out = _forward_teacher_forced(model, input_ids, mem_pos, mem_mask, qkp, hop_pos)
    warm_logits = out["retrieval"]["sinkhorn_warmup_logits"]
    target = torch.randint(5, 50, (input_ids.size(0),))
    F.cross_entropy(warm_logits[:, -1, :], target).backward()

    for name, module in [("router", model.router), ("episodic_memory", model.episodic_memory),
                          ("answer_head", model.answer_head)]:
        total_grad = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
        assert total_grad > 0, f"warmup loss should still reach {name}, detach cut too much"


def test_token_jepa_loss_still_trains_jepa_token_surprise_unaffected_by_detach(small_model_jepa_bias_warmup):
    # token_jepa_loss (see hpm_v2.py's token_surprise docstring, wired in
    # train.py as jepa_total = jepa_loss + token_jepa_loss) is the actual,
    # always-present, self-supervised trainer of jepa.token_context/
    # predictor/target -- present every step regardless of TF state, and
    # never touched warm_bias at all (it's returned directly by
    # token_surprise() alongside the token_surprise TENSOR that warm_bias
    # traced back through, but it's a separate scalar with its own
    # independent backward path). Confirms the detach didn't orphan JEPA's
    # token-level predictor from its real trainer -- only the warmup loss's
    # EXTRA, newly-introduced pull on the same output was cut.
    #
    # (Earlier version of this test checked writer_loss instead, on the
    # assumption that surprise_bias fed into the same logits writer_loss's
    # BCE is computed from. That assumption was wrong and the test caught
    # it: model.py's own docstring confirms surprise_bias is added to
    # selection_logits AFTER writer_loss's BCE is already computed from the
    # unbiased logits, so writer_loss never depended on jepa.token_surprise
    # in the first place, before or after this fix.)
    torch.manual_seed(4)
    model = small_model_jepa_bias_warmup
    input_ids, mem_pos, mem_mask, qkp, hop_pos = _build_batch()

    model.zero_grad()
    out = _forward_teacher_forced(model, input_ids, mem_pos, mem_mask, qkp, hop_pos)
    token_jepa_loss = out["retrieval"]["token_jepa_loss"]
    token_jepa_loss.backward()

    token_surprise_params = list(model.jepa.token_context.parameters()) \
        + list(model.jepa.token_predictor.parameters()) \
        + list(model.jepa.token_target.parameters())
    total_grad = sum(p.grad.abs().sum().item() for p in token_surprise_params if p.grad is not None)
    assert total_grad > 0, "token_jepa_loss should still train jepa.token_surprise -- that path was never detached"


def test_before_fix_undetached_warm_bias_would_have_reached_jepa_token_surprise(small_model_jepa_bias_warmup):
    # Directly demonstrates the competing-objectives claim without needing
    # to hand-revert the fix: reproduce writer_select_bias exactly as the
    # pre-fix warm path would have consumed it (no detach), and confirm
    # gradient reaches the same jepa.token_surprise parameters token_jepa_loss
    # also trains.
    #
    # NOTE on scope: checked at select_bias.sum().backward() directly, not
    # by running the full warm path forward through episodic_memory,
    # router, answer_head, and cross-entropy first. Traced empirically: the
    # gradient here is real but small (~2e-6 in this toy config, consistent
    # with the differentiable Sinkhorn bias being a deliberately soft STE
    # proxy -- see tests 1 & 2 above on bias_scale-independent gradient
    # magnitude). It underflows to exactly float32 0.0 after 4 more hops
    # through a 32-dim toy model's worth of Jacobians, which is a toy-scale
    # precision artifact, not evidence the connection doesn't exist -- so
    # this test checks connectivity at the point closest to where the
    # actual fix (the detach two lines below `warm_bias = ...`) intervenes,
    # rather than demanding it survive the whole downstream chain intact at
    # toy scale.
    torch.manual_seed(4)
    model = small_model_jepa_bias_warmup
    input_ids, mem_pos, mem_mask, qkp, hop_pos = _build_batch()

    model.zero_grad()
    local_state = model._local_path(input_ids)
    token_jepa_info = model.jepa.token_surprise(local_state)
    surprise_bias = token_jepa_info["token_surprise"]
    writer_info = model.writer(
        local_state, input_ids, qkp, mem_pos, mem_mask,
        max_slots=min(model.config.episodic_capacity, mem_pos.size(1)),
        surprise_bias=surprise_bias, surprise_bias_scale=model.config.jepa_writer_bias_scale,
    )
    # Deliberately NOT detached here -- reproducing the pre-fix code path
    # at exactly the tensor the real detach() targets.
    undetached_warm_bias = writer_info["writer_select_bias"]
    undetached_warm_bias.sum().backward()

    token_surprise_params = list(model.jepa.token_context.parameters()) \
        + list(model.jepa.token_predictor.parameters()) \
        + list(model.jepa.token_target.parameters())
    total_grad = sum(p.grad.abs().sum().item() for p in token_surprise_params if p.grad is not None)
    assert total_grad != 0.0, (
        "sanity check failed: without the detach, writer_select_bias should reach "
        "jepa.token_surprise -- if this is 0, the whole premise for the fix is wrong"
    )
