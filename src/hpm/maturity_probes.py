"""Small autograd probes shared by HPM maturity audits."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence
import torch


def jacobian_frobenius_norm(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> float:
    """Exact Frobenius norm of dy/dx for a small diagnostic tensor."""
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    if not torch.isfinite(y).all():
        raise ValueError("operator produced non-finite output")
    flat = y.reshape(-1)
    sq = x.new_zeros(())
    for i in range(flat.numel()):
        grad = torch.autograd.grad(flat[i], x, retain_graph=i + 1 < flat.numel(), allow_unused=False)[0]
        sq = sq + grad.square().sum()
    return float(torch.sqrt(sq).item())


def parameter_grad_norm(loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter], retain_graph: bool = False) -> float:
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    total = loss.new_zeros(())
    for g in grads:
        if g is not None:
            total = total + g.square().sum()
    return float(torch.sqrt(total).item())
