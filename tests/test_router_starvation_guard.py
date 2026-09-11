from types import SimpleNamespace

import pytest

from hpm.maturity_contracts import (
    BASELINE_ROUTER_STARVATION_GUARD,
    RouterStarvationError,
    RouterStarvationGuardConfig,
    enforce_router_starvation_guard,
)
from hpm.train import run_training


TEST_CONFIG = RouterStarvationGuardConfig(
    num_paths=4,
    initial_adverse_gap=2.0,
    max_adverse_gap_drift_per_step=0.0,
    discovery_horizon=10,
    minimum_probability=0.01,
)


def test_router_starvation_guard_aborts_artificially_starved_path():
    with pytest.raises(RouterStarvationError, match="router starvation guard aborted training"):
        enforce_router_starvation_guard(
            [0.70, 0.20, 0.0995, 0.0005],
            step=3,
            config=TEST_CONFIG,
            path_names=("local", "recurrent", "fast_weight", "episodic"),
        )


def test_router_starvation_guard_stays_silent_on_normal_distribution():
    result = enforce_router_starvation_guard(
        [0.28, 0.27, 0.24, 0.21],
        step=3,
        config=TEST_CONFIG,
        path_names=("local", "recurrent", "fast_weight", "episodic"),
    )
    assert result["active"] is True
    assert result["path_probabilities"]["episodic"] == pytest.approx(0.21)


def test_router_starvation_guard_aborts_on_dynamic_bound_even_above_p_min():
    # At A0=2,dmax=0,K=4 the dynamic bound is ~0.043; 0.02 is above the
    # absolute p_min=0.01 but still violates the finite-horizon bound.
    with pytest.raises(RouterStarvationError, match="below_dynamic_bound"):
        enforce_router_starvation_guard(
            [0.70, 0.20, 0.08, 0.02],
            step=3,
            config=TEST_CONFIG,
            path_names=("local", "recurrent", "fast_weight", "episodic"),
        )


def test_router_starvation_guard_stops_monitoring_after_discovery_horizon():
    result = enforce_router_starvation_guard(
        [0.9997, 0.0001, 0.0001, 0.0001],
        step=11,
        config=TEST_CONFIG,
        path_names=("local", "recurrent", "fast_weight", "episodic"),
    )
    assert result["active"] is False


def test_run_training_executes_enabled_router_starvation_guard(tmp_path):
    # Use the real empirical baseline calibration so this doubles as a normal
    # tiny-run smoke of the defaults intended for the first training run.
    metrics = run_training(
        SimpleNamespace(
            model="hpm_lite_v2",
            task="kv",
            seq_len=64,
            window=16,
            batch_size=2,
            steps=2,
            eval_every=2,
            eval_batches=1,
            d_model=32,
            layers=1,
            heads=4,
            lr=3.0e-4,
            seed=321,
            device="cpu",
            lambda_ret=0.1,
            lambda_writer=0.1,
            learned_writer_teacher_forcing_steps=1,
            top_k=1,
            memory_null_slot=True,
            null_score_init=0.0,
            memory_control="normal",
            write_mode="learned",
            oracle_memory=True,
            num_facts=4,
            repeated_keys=False,
            similar_values=False,
            distractor_fact_spans=0,
            query_key_noise_only=False,
            fact_order="random",
            out_dir=str(tmp_path),
            save_checkpoint=False,
            log_every=2,
            save_step_log=False,
            record_vram=False,
            router_starvation_guard=True,
            router_starvation_a0=BASELINE_ROUTER_STARVATION_GUARD.initial_adverse_gap,
            router_starvation_d_max=BASELINE_ROUTER_STARVATION_GUARD.max_adverse_gap_drift_per_step,
            router_discovery_horizon=BASELINE_ROUTER_STARVATION_GUARD.discovery_horizon,
            router_p_min=BASELINE_ROUTER_STARVATION_GUARD.minimum_probability,
        )
    )
    assert metrics["router_starvation_guard"] is True
    assert metrics["router_starvation_guard_last"]["step"] == 2
    assert metrics["router_starvation_guard_last"]["active"] is True
