import pytest
import torch

from hpm.writer_objectives import (
    capped_sigmoid_topk_membership,
    topc_required_margin_diagnostics,
    topc_set_coverage_loss,
)


def _inputs(positive_scores: float, negative_scores: float):
    logits = torch.full((1, 8), negative_scores, requires_grad=True)
    with torch.no_grad():
        logits[0, :2] = positive_scores
    labels = torch.zeros(1, 8)
    labels[0, :2] = 1.0
    valid = torch.ones(1, 8, dtype=torch.bool)
    return logits, labels, valid


def test_topc_set_coverage_prefers_a_complete_required_set_and_has_ranking_gradients():
    good_logits, labels, valid = _inputs(positive_scores=1.0, negative_scores=-1.0)
    bad_logits, _, _ = _inputs(positive_scores=-1.0, negative_scores=1.0)

    good_loss, good_info = topc_set_coverage_loss(good_logits, labels, valid, capacity=2)
    bad_loss, _ = topc_set_coverage_loss(bad_logits, labels, valid, capacity=2)
    bad_loss.backward()

    assert good_loss < bad_loss
    assert good_info["max_membership_mass_error"] <= 1.0e-3
    # Raising a required score must lower the loss; raising a distractor must
    # raise it.  This is the direct Top-C ranking signal absent from BCE.
    assert torch.all(bad_logits.grad[0, :2] < 0.0)
    assert torch.all(bad_logits.grad[0, 2:] > 0.0)


def test_topc_set_coverage_is_permutation_equivariant():
    torch.manual_seed(14)
    logits = torch.randn(1, 16)
    labels = torch.zeros(1, 16)
    labels[0, [1, 7, 11]] = 1.0
    valid = torch.ones(1, 16, dtype=torch.bool)
    valid[0, [4, 15]] = False

    original, _ = topc_set_coverage_loss(logits, labels, valid, capacity=4)
    perm = torch.randperm(16)
    permuted, _ = topc_set_coverage_loss(logits[:, perm], labels[:, perm], valid[:, perm], capacity=4)

    assert torch.allclose(original, permuted, atol=1.0e-6, rtol=0.0)


@pytest.mark.parametrize("n,k,scale", [(3, 1, 0.0), (5, 4, 1.0e4), (32, 4, 1.0), (512, 16, 1.0e4)])
def test_capped_sigmoid_topk_preserves_budget_at_extreme_score_scales(n, k, scale):
    torch.manual_seed(n * 100 + k)
    scores = (torch.randn(n) * scale).requires_grad_(True)
    membership = capped_sigmoid_topk_membership(scores, k=k, temperature=0.3, n_iters=48)
    membership[0].backward()

    assert torch.isfinite(membership).all()
    assert torch.isfinite(scores.grad).all()
    assert membership.min() >= 0.0
    assert membership.max() <= 1.0
    assert abs(float(membership.sum().detach()) - k) <= 1.0e-4


def test_capped_sigmoid_topk_implicit_gradient_matches_finite_difference_away_from_range_extrema():
    # Scores at 0 and 5 fix the detached range; probe index 1 so a finite
    # difference observes the same local normalized-score geometry as the
    # custom implicit backward.
    scores = torch.tensor([-1.0, 0.20, 0.10, -0.15, 0.5, 1.0], requires_grad=True)
    weights = torch.tensor([0.0, 1.0, -0.7, 0.3, 0.0, 0.0])
    output = capped_sigmoid_topk_membership(scores, k=2, temperature=0.3, n_iters=48)
    scalar = (weights * output).sum()
    scalar.backward()
    analytic = float(scores.grad[1])

    delta = 1.0e-3
    with torch.no_grad():
        plus = scores.detach().clone(); plus[1] += delta
        minus = scores.detach().clone(); minus[1] -= delta
        plus_value = float((weights * capped_sigmoid_topk_membership(plus, k=2, temperature=0.3)).sum())
        minus_value = float((weights * capped_sigmoid_topk_membership(minus, k=2, temperature=0.3)).sum())
    numerical = (plus_value - minus_value) / (2.0 * delta)

    assert abs(analytic - numerical) < 2.0e-3


def test_topc_set_coverage_corrects_an_adversarial_hard_ranking():
    # The two required candidates start below every distractor, so hard Top-2
    # coverage is initially zero.  This is a direct end-to-end test of the
    # objective's intended ranking signal, independent of the HPM backbone.
    logits = torch.full((1, 16), 1.0, requires_grad=True)
    with torch.no_grad():
        logits[0, :2] = -1.0
    labels = torch.zeros(1, 16)
    labels[0, :2] = 1.0
    valid = torch.ones(1, 16, dtype=torch.bool)
    optimizer = torch.optim.Adam([logits], lr=0.1)

    initial = int(labels.gather(1, logits.detach().topk(2, dim=-1).indices).sum().item())
    for _ in range(80):
        optimizer.zero_grad()
        loss, _ = topc_set_coverage_loss(logits, labels, valid, capacity=2)
        loss.backward()
        optimizer.step()
    final = int(labels.gather(1, logits.detach().topk(2, dim=-1).indices).sum().item())

    assert initial == 0
    assert final == 2


def test_topc_margin_is_the_exact_continuous_boundary_for_all_required_selection():
    # C=p=2: the boundary is the highest distractor. The first row is fully
    # retained; the second loses both required entries to a distractor.
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0], [1.0, 5.0, 0.5, 0.0]])
    labels = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    selected = torch.ones(2, 2, dtype=torch.bool)

    info = topc_required_margin_diagnostics(logits, labels, valid, selected)

    # min required - max distractor = 4-1 and .5-5, respectively.
    assert info["samples"] == 2.0
    assert info["mean_required_topc_margin"] == pytest.approx(-0.75)
    assert info["all_required_topc_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "labels, capacity, message",
    [
        (torch.zeros(1, 8), 2, "no required candidates"),
        (torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 2, "requires 3 candidates"),
    ],
)
def test_topc_set_coverage_rejects_invalid_capacity_contracts(labels, capacity, message):
    logits = torch.zeros(1, 8)
    valid = torch.ones(1, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match=message):
        topc_set_coverage_loss(logits, labels, valid, capacity=capacity)


def test_topc_set_coverage_rejects_nonselective_all_fit_setup():
    logits = torch.zeros(1, 2)
    labels = torch.tensor([[1.0, 0.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="every candidate fits"):
        topc_set_coverage_loss(logits, labels, valid, capacity=2)
