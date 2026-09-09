"""Tests for the frozen-router causal intervention semantics.

Two layers:

1. Direct ``HpmV2PathRouter.forward`` units with hand-made path tensors.
   These prove the property the whole diagnostic rests on: the contribution
   mask is applied AFTER softmax and NEVER renormalized, so zeroing a path
   removes exactly ``weight_k * path_k`` from the mixture -- the gate cannot
   re-route around the ablation. A regression that renormalizes survivors
   (or recomputes weights from ablated states) would silently turn the
   necessity test into a different architecture; the "not equal to the
   renormalized mixture" assertion catches exactly that.

2. Model-level behavior through ``HpmLiteV2Model.forward``: freezing is
   semantics-preserving (replaying captured weights reproduces the normal
   pass bit-for-bit), interventions refuse on episodic_only models (no
   router exists there), and invalid inputs are rejected loudly rather than
   silently mis-weighting an evaluation.
"""
import torch

from hpm.hpm_v2 import HpmV2PathRouter
from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

import pytest


def _router_and_paths(d_model=8, bsz=2, seq_len=5, seed=0):
    torch.manual_seed(seed)
    router = HpmV2PathRouter(d_model)
    paths = [torch.randn(bsz, seq_len, d_model) for _ in range(4)]
    return router, paths


def _manual_mixture(weights, paths, keep=range(4)):
    return sum(weights[..., i : i + 1] * path for i, path in enumerate(paths) if i in keep)


def test_mask_all_ones_equals_plain_weighted_sum():
    router, paths = _router_and_paths()
    fixed = torch.softmax(torch.randn(2, 5, 4), dim=-1)

    mixed_no_mask, _, _ = router(*paths, fixed_weights=fixed)
    assert torch.allclose(mixed_no_mask, _manual_mixture(fixed, paths), atol=1e-6)

    mixed_ones_mask, _, _ = router(
        *paths, fixed_weights=fixed, contribution_mask=torch.ones(4)
    )
    assert torch.allclose(mixed_ones_mask, mixed_no_mask, atol=1e-7), (
        "an explicit all-ones mask must be identical to no mask at all"
    )


def test_zeroed_path_is_removed_without_renormalization():
    router, paths = _router_and_paths(seed=3)
    fixed = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    k = 2  # fast_weight slot in ROUTER_PATH_NAMES order
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0])

    mixed, _, _ = router(*paths, fixed_weights=fixed, contribution_mask=mask)
    assert torch.allclose(mixed, _manual_mixture(fixed, paths, keep={0, 1, 3}), atol=1e-6), (
        "zeroing path k must remove exactly weight_k * path_k -- nothing more, nothing less"
    )

    # The anti-regression guard: if someone ever renormalizes the surviving
    # weights (the classic 'helpful' fix that destroys the necessity test),
    # the mixture scales up by ~1/(1-w_k) and this assertion starts failing.
    renorm = fixed.clone()
    renorm[..., k] = 0.0
    renorm = renorm / renorm.sum(dim=-1, keepdim=True)
    renormalized_mixture = _manual_mixture(renorm, paths, keep={0, 1, 3})
    assert not torch.allclose(mixed, renormalized_mixture, atol=1e-4), (
        "masked output matches a RENORMALIZED mixture -- post-gate no-renorm semantics are broken"
    )


def test_refreezing_learned_weights_reproduces_learned_pass():
    router, paths = _router_and_paths(seed=11)
    mixed_learned, learned_weights, logits = router(*paths)
    assert torch.allclose(learned_weights, torch.softmax(logits, dim=-1), atol=1e-6)

    mixed_refrozen, _, _ = router(*paths, fixed_weights=learned_weights.detach())
    assert torch.allclose(mixed_learned, mixed_refrozen, atol=1e-6), (
        "freezing the gate's own current weights must not change the forward pass"
    )


def test_router_rejects_invalid_interventions():
    router, paths = _router_and_paths()
    good = torch.softmax(torch.randn(2, 5, 4), dim=-1)

    with pytest.raises(ValueError, match="shape"):
        router(*paths, fixed_weights=torch.softmax(torch.randn(2, 5, 3), dim=-1))
    bad_sum = good.clone()
    bad_sum[..., 0] += 0.5
    with pytest.raises(ValueError, match="sum to one"):
        router(*paths, fixed_weights=bad_sum)
    with pytest.raises(ValueError, match="finite and non-negative"):
        router(*paths, fixed_weights=-good.abs() / good.sum(dim=-1, keepdim=True))
    nonfinite = good.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        router(*paths, fixed_weights=nonfinite)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        router(*paths, contribution_mask=torch.tensor([1.0, 1.0, 1.0, 1.5]))
    with pytest.raises(ValueError, match="broadcast"):
        router(*paths, contribution_mask=torch.ones(2, 5, 9))


def _tiny_model(episodic_only=False):
    torch.manual_seed(0)
    cfg = HpmLiteV2Config(d_model=16, layers=1, heads=2, window=8, block_size=4, episodic_only=episodic_only)
    return HpmLiteV2Model(cfg)


def _kv_batch(bsz=2, seq_len=24):
    input_ids = torch.randint(5, 50, (bsz, seq_len))
    starts = torch.linspace(0, seq_len - 6, 4).long()
    mem_pos = torch.stack([starts, starts + 1], dim=-1)[None].expand(bsz, 4, 2).contiguous()
    mem_mask = torch.ones(bsz, 4, dtype=torch.bool)
    qkp = torch.full((bsz,), seq_len - 2, dtype=torch.long)
    input_ids[:, -2] = 1
    input_ids[:, -1] = 2
    return input_ids, mem_pos, mem_mask, qkp


def test_model_freeze_equivalence_full_pass():
    model = _tiny_model()
    model.eval()
    input_ids, mem_pos, mem_mask, qkp = _kv_batch()
    kwargs = dict(
        memory_token_positions=mem_pos,
        memory_mask=mem_mask,
        answer_positions=None,
        query_key_positions=qkp,
        top_k=1,
        task="kv",
    )
    with torch.no_grad():
        normal = model(input_ids, **kwargs)
        frozen = model(
            input_ids,
            router_fixed_weights=normal["retrieval"]["router_weights"],
            **kwargs,
        )
    assert torch.allclose(normal["logits"], frozen["logits"], atol=1e-6), (
        "replaying the captured per-token router weights must reproduce the normal pass exactly"
    )


def test_model_refuses_interventions_under_episodic_only():
    model = _tiny_model(episodic_only=True)
    model.eval()
    input_ids, mem_pos, mem_mask, qkp = _kv_batch()
    with pytest.raises(ValueError, match="episodic_only=False"):
        model(
            input_ids,
            memory_token_positions=mem_pos,
            memory_mask=mem_mask,
            answer_positions=None,
            query_key_positions=qkp,
            top_k=1,
            task="kv",
            router_contribution_mask=torch.ones(4),
        )
