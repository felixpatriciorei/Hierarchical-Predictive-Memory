import torch
import torch.nn.functional as F
import pytest
from hpm.capacity_theory import orthogonal_delta_exact_capacity,delta_rule_recall,normalized_recall_cosine,overwrite_cosine

def test_orthogonal_exact_storage():
    d=8
    keys=torch.eye(d)
    values=F.normalize(torch.randn(d,d,generator=torch.Generator().manual_seed(1)),dim=-1)
    pred=delta_rule_recall(keys,values)
    assert torch.allclose(pred,values,atol=1e-6,rtol=1e-6)
    assert orthogonal_delta_exact_capacity(d)==d

def test_rebinding_same_key_overwrites():
    g=torch.Generator().manual_seed(2)
    key=torch.randn(16,generator=g)
    old=F.normalize(torch.randn(16,generator=g),dim=-1)
    new=F.normalize(torch.randn(16,generator=g),dim=-1)
    to_new,to_old=overwrite_cosine(key,old,new)
    assert to_new > 0.999
    assert to_new > to_old

def test_random_interference_not_claimed_exact():
    g=torch.Generator().manual_seed(3)
    keys=F.normalize(torch.randn(32,8,generator=g),dim=-1)
    vals=F.normalize(torch.randn(32,8,generator=g),dim=-1)
    score=normalized_recall_cosine(keys,vals)
    assert -1 <= score <= 1
