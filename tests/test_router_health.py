from __future__ import annotations

import math

import pytest
import torch

from hpm.hpm_v2 import HpmV2PathRouter
from hpm.metrics import router_health_metrics


def _paths(*, batch: int = 2, seq: int = 3, d_model: int = 5, num_paths: int = 4):
    gen = torch.Generator().manual_seed(1234)
    return tuple(torch.randn(batch, seq, d_model, generator=gen) for _ in range(num_paths))


def test_forward_contract_is_unchanged_and_diagnostics_are_noninvasive():
    router = HpmV2PathRouter(d_model=5, num_paths=4, logit_clamp=None)
    paths = _paths(d_model=5)

    mixed, weights, logits = router(*paths)
    mixed_d, weights_d, logits_d, diagnostics = router.route_with_diagnostics(*paths)

    assert torch.equal(mixed, mixed_d)
    assert torch.equal(weights, weights_d)
    assert torch.equal(logits, logits_d)
    assert torch.equal(diagnostics["raw_logits"], logits)
    assert torch.equal(diagnostics["effective_logits"], logits)
    assert torch.equal(diagnostics["clamp_jacobian"], torch.ones_like(logits))
    health = router_health_metrics(weights, logits, router_raw_logits=diagnostics["raw_logits"], router_clamp_jacobian=diagnostics["clamp_jacobian"])
    assert health["router_clamp_attenuation_ratio_mean"].item() == pytest.approx(1.0)


def test_fixed_weight_post_gate_intervention_semantics_are_unchanged():
    router = HpmV2PathRouter(d_model=3, num_paths=4)
    paths = _paths(batch=1, seq=2, d_model=3)
    fixed = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]])
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])

    mixed, weights, _ = router(*paths, fixed_weights=fixed, contribution_mask=mask)
    expected = sum(fixed[..., i : i + 1] * mask[i] * paths[i] for i in range(4))

    assert torch.equal(weights, fixed)
    assert torch.allclose(mixed, expected)


def test_clamp_guarantees_forward_probability_bound_but_exposes_raw_lock():
    clamp = 3.0
    router = HpmV2PathRouter(d_model=2, num_paths=4, logit_clamp=clamp)
    with torch.no_grad():
        router.proj.weight.zero_()
        router.proj.bias.copy_(torch.tensor([100.0, -100.0, -100.0, -100.0]))

    paths = tuple(torch.zeros(1, 1, 2) for _ in range(4))
    _, weights, logits, diagnostics = router.route_with_diagnostics(*paths)

    theoretical_max = 1.0 / (1.0 + 3.0 * math.exp(-2.0 * clamp))
    assert logits.abs().max().item() <= clamp + 1.0e-7
    assert weights.max().item() <= theoretical_max + 1.0e-6
    assert diagnostics["raw_logits"].abs().max().item() == pytest.approx(100.0)
    # tanh has saturated even though softmax remains strictly inside the simplex.
    assert diagnostics["clamp_jacobian"].max().item() < 1.0e-6

    health = router_health_metrics(
        weights,
        logits,
        router_raw_logits=diagnostics["raw_logits"],
        router_clamp_jacobian=diagnostics["clamp_jacobian"],
    )
    assert health["router_softmax_jacobian_fro_mean"].item() > 0.0
    assert health["router_raw_to_weight_jacobian_fro_mean"].item() < 1.0e-6
    assert health["router_clamp_attenuation_ratio_mean"].item() < 1.0e-6
    assert health["router_raw_logit_abs_mean"].item() > health["router_logit_abs_mean"].item() * 20.0


def test_raw_to_weight_jacobian_metric_matches_autograd():
    clamp = 3.0
    raw = torch.tensor([0.7, -0.2, 1.1, -0.5], dtype=torch.float64, requires_grad=True)

    def f(a: torch.Tensor) -> torch.Tensor:
        z = clamp * torch.tanh(a / clamp)
        return torch.softmax(z, dim=-1)

    weights = f(raw)
    effective = clamp * torch.tanh(raw / clamp)
    clamp_jacobian = 1.0 - torch.tanh(raw / clamp).square()
    health = router_health_metrics(
        weights.view(1, 1, -1),
        effective.view(1, 1, -1),
        router_raw_logits=raw.view(1, 1, -1),
        router_clamp_jacobian=clamp_jacobian.view(1, 1, -1),
    )

    jac = torch.autograd.functional.jacobian(f, raw)
    expected_fro = jac.square().sum().sqrt()
    assert torch.allclose(
        health["router_raw_to_weight_jacobian_fro_mean"],
        expected_fro,
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_health_metric_detects_clamp_lock_separately_from_softmax_saturation():
    clamp = 3.0

    def health_for(raw_values):
        raw = torch.tensor(raw_values, dtype=torch.float64).view(1, 1, 4)
        effective = clamp * torch.tanh(raw / clamp)
        derivative = 1.0 - torch.tanh(raw / clamp).square()
        weights = torch.softmax(effective, dim=-1)
        return router_health_metrics(
            weights,
            effective,
            router_raw_logits=raw,
            router_clamp_jacobian=derivative,
        )

    moderate = health_for([2.0, -2.0, -2.0, -2.0])
    locked = health_for([30.0, -30.0, -30.0, -30.0])

    # The forward probabilities are bounded in both cases, but raw->weight
    # sensitivity collapses once the tanh itself saturates.
    assert locked["router_weight_max_mean"].item() < 1.0
    assert locked["router_softmax_jacobian_fro_mean"].item() > 0.0
    assert locked["router_clamp_jacobian_mean"].item() < moderate["router_clamp_jacobian_mean"].item() * 1.0e-6
    assert locked["router_raw_to_weight_jacobian_fro_mean"].item() < moderate[
        "router_raw_to_weight_jacobian_fro_mean"
    ].item() * 1.0e-6


def test_invalid_clamp_is_rejected():
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            HpmV2PathRouter(d_model=4, logit_clamp=value)
