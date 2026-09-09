import math
import pytest
from hpm.maturity_contracts import (
    router_probability_floor,max_router_gap_drift_for_floor,write_temperature_step_bound,
    hard_selection_safe,episodic_capacity_feasible,jepa_stage_isolation_ok,finite_gradient_ratio,
)

def test_router_floor_and_inverse_are_consistent():
    pmin=1e-3; k=4; a0=0.5; t=100
    d=max_router_gap_drift_for_floor(k,a0,t,pmin)
    p=router_probability_floor(k,a0,d,t)
    assert p == pytest.approx(pmin, rel=1e-10)

def test_router_floor_decreases_with_horizon():
    assert router_probability_floor(4,0.2,0.01,100) < router_probability_floor(4,0.2,0.01,10)

def test_write_step_bound():
    assert write_temperature_step_bound(2.0,0.1)==pytest.approx(0.05)
    assert math.isinf(write_temperature_step_bound(0.0,0.1))

def test_hard_selection_guard():
    assert hard_selection_safe(0.2,0.09)
    assert not hard_selection_safe(0.2,0.1)

def test_capacity_hard_constraint():
    assert episodic_capacity_feasible(8,8)
    assert not episodic_capacity_feasible(9,8)

def test_jepa_stages():
    assert jepa_stage_isolation_ok(lambda_jepa=0,use_token_jepa_aux=False,use_jepa_writer_bias=False,stage=0)
    assert jepa_stage_isolation_ok(lambda_jepa=.1,use_token_jepa_aux=False,use_jepa_writer_bias=False,stage=1)
    assert jepa_stage_isolation_ok(lambda_jepa=.1,use_token_jepa_aux=True,use_jepa_writer_bias=False,stage=2)
    assert jepa_stage_isolation_ok(lambda_jepa=.1,use_token_jepa_aux=True,use_jepa_writer_bias=True,stage=3)

def test_gradient_ratio():
    assert finite_gradient_ratio(1,4)==pytest.approx(.25)
