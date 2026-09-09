import torch

from hpm.data import VOCAB_SIZE
from hpm.hpm_v2 import (
    BlockwiseSelectiveRecurrentState,
    FastWeightBlockMemory,
    HpmV2PathRouter,
    JepaLiteAuxiliary,
    block_summaries,
)
from hpm.metrics import router_z_loss
from hpm.model import SupervisedMemoryWriter
from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model


def test_blockwise_selective_state_shape_and_causality():
    torch.manual_seed(0)
    module = BlockwiseSelectiveRecurrentState(d_model=16, block_size=4)
    x = torch.randn(2, 12, 16)
    y = module(x)
    assert y.shape == x.shape
    # First block sees zero prior recurrent state before output projection.
    # The projection bias can make it nonzero, but all tokens in the first block
    # receive the same prior state and therefore same recurrent output.
    assert torch.allclose(y[:, 0, :], y[:, 3, :], atol=1e-5)
    assert not torch.allclose(y[:, 0, :], y[:, 4, :])


def test_fast_weight_block_memory_shape_and_gradients():
    torch.manual_seed(1)
    module = FastWeightBlockMemory(d_model=12, block_size=3)
    x = torch.randn(2, 9, 12, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_router_mixes_four_paths():
    router = HpmV2PathRouter(d_model=8, num_paths=4)
    paths = [torch.randn(2, 5, 8) for _ in range(4)]
    mixed, weights, logits = router(*paths)
    assert mixed.shape == paths[0].shape
    assert weights.shape == (2, 5, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert logits.shape == (2, 5, 4)
    assert torch.allclose(torch.softmax(logits, dim=-1), weights, atol=1e-6)


def test_router_z_loss_penalizes_logit_magnitude_not_confidence():
    """Pins the exact failure mode from the real sweep: router_logit_abs_mean
    sat flat at ~1.2-1.7 through step 200, then jumped 3-7x to 4.8-9.2 in a
    single eval window right when the router locked one-hot and never
    recovered. z-loss must penalize that magnitude growth directly, and
    must NOT penalize a well-separated-but-modest-scale one-hot decision
    the same way router_entropy_loss would -- see metrics.router_z_loss's
    docstring for why entropy alone can't distinguish these two cases.
    """
    torch.manual_seed(0)
    small_scale = torch.tensor([[[3.0, -3.0, -3.0, -3.0]]])  # modest, well-separated
    large_scale = torch.tensor([[[30.0, -30.0, -30.0, -30.0]]])  # same softmax output, blown-up logits

    # Both saturate softmax to (near-)identical one-hot weights...
    assert torch.allclose(torch.softmax(small_scale, dim=-1), torch.softmax(large_scale, dim=-1), atol=1e-2)

    # ...but z-loss, unlike entropy, tells them apart: large-magnitude logits
    # are penalized far more than modest, equally-confident ones.
    small_z = router_z_loss(small_scale)
    large_z = router_z_loss(large_scale)
    assert large_z.item() > small_z.item() * 10

    # Sanity: the flat pre-collapse magnitude (~1.2-1.7) from the real sweep
    # should sit near-zero on this scale; the post-collapse magnitude
    # (~4.8-9.2) should be clearly, substantially larger.
    pre_collapse = torch.full((1, 1, 4), 1.5)
    post_collapse = torch.full((1, 1, 4), 8.0)
    assert router_z_loss(post_collapse).item() > router_z_loss(pre_collapse).item() * 5


def test_router_z_loss_gradient_shrinks_large_logits():
    """The actual mechanism that should prevent the step-250-style trap:
    gradient descent on router_z_loss must reduce logit magnitude, giving
    softmax a way back out of a near-saturated corner instead of the
    vanishing-gradient dead end that caused the real collapse to never
    recover through step 600."""
    logits = torch.tensor([[[15.0, -15.0, -15.0, -15.0]]], requires_grad=True)
    loss = router_z_loss(logits)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    with torch.no_grad():
        updated = logits - 0.1 * logits.grad
    assert updated.abs().mean().item() < logits.abs().mean().item()



def test_router_logit_clamp_bounds_magnitude_regardless_of_raw_scale():
    """Pins the actual fix after lambda_router_z_loss=1e-3 was tested on a
    real sweep and did NOT prevent collapse (router_logit_abs_mean still hit
    4.2-12.9, comparable to the unregularized baseline, because a soft
    penalty's effectiveness depends on outweighing whatever else is in the
    loss at that step -- observed router_z_loss up to 129 contributed
    1e-3*129=0.129 against a total loss of ~21, under 1% resistance).

    logit_clamp is an architectural bound, not an incentive: |logit| must
    stay <= clamp UNCONDITIONALLY, even when the router's raw linear
    projection produces values orders of magnitude larger than the clamp --
    there is no lambda to outrace, because the bound isn't a training
    target, it's a forward-pass computation.
    """
    torch.manual_seed(0)
    router = HpmV2PathRouter(d_model=8, num_paths=4, logit_clamp=3.0)
    # Force the router's own weights to be huge, mimicking the unbounded
    # growth actually observed at step 250-600 in the real sweep (raw
    # logit_abs_mean reached ~15 there).
    with torch.no_grad():
        router.proj.weight.mul_(1000.0)
        router.proj.bias.mul_(1000.0)
    paths = [torch.randn(2, 5, 8) for _ in range(4)]
    mixed, weights, logits = router(*paths)
    assert logits.abs().max().item() <= 3.0 + 1e-4
    # Never a literal, permanent one-hot -- some gradient must always remain.
    assert weights.max().item() < 0.999
    assert torch.isfinite(mixed).all()


def test_router_logit_clamp_default_off_is_byte_identical_to_before():
    """logit_clamp=None (the default) must reproduce the exact pre-existing
    unclamped behavior -- this flag must not silently change results for
    any run that doesn't explicitly opt in."""
    torch.manual_seed(1)
    paths = [torch.randn(2, 5, 8) for _ in range(4)]

    torch.manual_seed(0)
    router_a = HpmV2PathRouter(d_model=8, num_paths=4)
    torch.manual_seed(0)
    router_b = HpmV2PathRouter(d_model=8, num_paths=4, logit_clamp=None)

    mixed_a, weights_a, logits_a = router_a(*paths)
    mixed_b, weights_b, logits_b = router_b(*paths)
    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(weights_a, weights_b)
    assert torch.allclose(mixed_a, mixed_b)


def _tiny_writer_inputs(batch: int = 2, seq_len: int = 6, d_model: int = 8):
    """Minimal, valid inputs for SupervisedMemoryWriter.forward: a fixed
    query at the last position, one oracle fact starting at position 0, and
    token ids that are never QUERY(4)/ANSWER(5) so the resulting valid mask
    is non-empty."""
    hidden = torch.randn(batch, seq_len, d_model)
    input_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    query_key_positions = torch.full((batch,), seq_len - 1, dtype=torch.long)
    oracle_memory_token_positions = torch.zeros(batch, 1, 2, dtype=torch.long)
    oracle_memory_mask = torch.ones(batch, 1)
    return hidden, input_ids, query_key_positions, oracle_memory_token_positions, oracle_memory_mask


def test_surprise_bias_clamp_bounds_magnitude_regardless_of_raw_scale():
    """Pins the actual fix after a same-seed jepa_bias on/off comparison
    (seed 0 of the primary-loss-anneal sweep) showed the unclamped surprise
    bias can dominate the oracle-trained selection logits enough to
    destabilize which candidates get chosen (elevated candidate entropy/KL,
    reduced top-k Jaccard overlap, non-converging sinkhorn_warmup_loss) for
    some seeds while being neutral-to-helpful for others.

    surprise_bias_clamp is an architectural bound, not an incentive: the
    bias's contribution to selection_logits must stay <= clamp in magnitude
    UNCONDITIONALLY, even when the raw surprise score is orders of magnitude
    larger than the clamp -- mirrors router_logit_clamp's own guarantee and
    its test (test_router_logit_clamp_bounds_magnitude_regardless_of_raw_scale)
    exactly."""
    torch.manual_seed(0)
    writer = SupervisedMemoryWriter(d_model=8)
    hidden, input_ids, query_key_positions, oracle_pos, oracle_mask = _tiny_writer_inputs()
    huge_surprise = torch.full((2, 6), 1000.0)

    baseline = writer(
        hidden, input_ids, query_key_positions, oracle_pos, oracle_mask, max_slots=2,
    )
    clamped = writer(
        hidden, input_ids, query_key_positions, oracle_pos, oracle_mask, max_slots=2,
        surprise_bias=huge_surprise, surprise_bias_scale=1.0, surprise_bias_clamp=3.0,
    )
    # writer_selection_logits = writer_logits + clamped_bias. With a huge raw
    # surprise score (1000.0) but clamp=3.0, the bias's contribution to the
    # valid positions must never exceed 3.0 in magnitude, regardless of how
    # large the raw surprise term is.
    valid = clamped["writer_valid_mask"]
    bias_contribution = (clamped["writer_selection_logits"] - clamped["writer_logits"])[valid]
    assert bias_contribution.abs().max().item() <= 3.0 + 1.0e-4
    assert torch.isfinite(clamped["writer_select_bias"]).all()
    assert torch.isfinite(clamped["writer_full_memory_token_positions"]).all()
    # writer_loss (the BCE term) must be identical to the unclamped baseline
    # -- surprise_bias, clamped or not, never touches the pre-BCE logits.
    assert torch.allclose(baseline["writer_loss"], clamped["writer_loss"])


def test_surprise_bias_clamp_default_off_is_byte_identical_to_before():
    """surprise_bias_clamp=None (the default) must reproduce the exact
    pre-existing unclamped surprise_bias behavior -- this flag must not
    silently change results for any run that doesn't explicitly opt in."""
    torch.manual_seed(2)
    hidden, input_ids, query_key_positions, oracle_pos, oracle_mask = _tiny_writer_inputs()
    surprise = torch.randn(2, 6)

    torch.manual_seed(0)
    writer_a = SupervisedMemoryWriter(d_model=8)
    torch.manual_seed(0)
    writer_b = SupervisedMemoryWriter(d_model=8)

    out_a = writer_a(
        hidden, input_ids, query_key_positions, oracle_pos, oracle_mask, max_slots=2,
        surprise_bias=surprise, surprise_bias_scale=1.0,
    )
    out_b = writer_b(
        hidden, input_ids, query_key_positions, oracle_pos, oracle_mask, max_slots=2,
        surprise_bias=surprise, surprise_bias_scale=1.0, surprise_bias_clamp=None,
    )
    assert torch.allclose(out_a["writer_select_bias"], out_b["writer_select_bias"])
    assert torch.allclose(out_a["writer_loss"], out_b["writer_loss"])


def test_jepa_lite_auxiliary_is_finite():
    torch.manual_seed(2)
    x = torch.randn(2, 16, 20)
    summaries = block_summaries(x, block_size=4)
    jepa = JepaLiteAuxiliary(d_model=20, latent_dim=10)
    info = jepa(summaries)
    assert set(info) == {"jepa_loss", "jepa_cosine", "jepa_target_std"}
    assert torch.isfinite(info["jepa_loss"])
    assert torch.isfinite(info["jepa_cosine"])
    assert info["jepa_target_std"] > 0


def test_token_jepa_auxiliary_can_train_without_controlling_writer():
    """Token JEPA must be independently trainable.

    The paired models have identical weights and inputs.  Enabling only the
    token-level auxiliary must expose its loss, but must not alter the
    writer's selection logits or the primary answer logits.  Its gradient is
    allowed to reach the local representation (that is the auxiliary's
    purpose), but not the writer parameters.
    """
    torch.manual_seed(17)
    base_config = HpmLiteV2Config(
        model_type="hpm_lite_v2", d_model=16, layers=1, heads=2, window=4,
        max_seq_len=16, use_learned_writer=True, use_jepa_aux=True,
        use_token_jepa_aux=False, use_jepa_writer_bias=False,
    )
    aux_config = HpmLiteV2Config(
        model_type="hpm_lite_v2", d_model=16, layers=1, heads=2, window=4,
        max_seq_len=16, use_learned_writer=True, use_jepa_aux=True,
        use_token_jepa_aux=True, use_jepa_writer_bias=False,
    )
    base = HpmLiteV2Model(base_config)
    aux = HpmLiteV2Model(aux_config)
    aux.load_state_dict(base.state_dict())

    input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    positions = torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    memory_mask = torch.ones(2, 2, dtype=torch.bool)
    answer_positions = torch.tensor([7, 7])
    query_positions = torch.tensor([6, 6])

    kwargs = dict(
        input_ids=input_ids,
        memory_token_positions=positions,
        memory_mask=memory_mask,
        answer_positions=answer_positions,
        query_key_positions=query_positions,
        use_learned_writer=True,
        learned_writer_teacher_forcing=False,
    )
    base_out = base(**kwargs)
    aux_out = aux(**kwargs)

    assert "token_jepa_loss" not in base_out["retrieval"]
    assert torch.isfinite(aux_out["retrieval"]["token_jepa_loss"])
    assert torch.allclose(base_out["logits"], aux_out["logits"])
    assert torch.allclose(
        base_out["retrieval"]["writer_selection_logits"],
        aux_out["retrieval"]["writer_selection_logits"],
    )

    aux_out["retrieval"]["token_jepa_loss"].backward()
    token_params = list(aux.jepa.token_context.parameters()) + list(aux.jepa.token_predictor.parameters())
    assert sum(p.grad.abs().sum().item() for p in token_params if p.grad is not None) > 0
    assert all(p.grad is None for p in aux.writer.parameters())

