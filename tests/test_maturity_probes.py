import torch
import pytest
from hpm.maturity_probes import jacobian_frobenius_norm

def test_jacobian_probe_matches_linear_map():
    A=torch.tensor([[1.,2.],[-3.,4.]])
    x=torch.tensor([.2,.3])
    got=jacobian_frobenius_norm(lambda z:A@z,x)
    assert got==pytest.approx(float(torch.linalg.vector_norm(A)),rel=1e-6)
