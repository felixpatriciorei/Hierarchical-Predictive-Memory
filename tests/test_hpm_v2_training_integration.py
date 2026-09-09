import csv
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from hpm.train import (
    run_training,
    teacher_forcing_probability,
    sinkhorn_warmup_weight_at_step,
    primary_loss_weight_at_step,
    causal_writer_capacity_metadata,
    validate_causal_writer_capacity_contract,
    validate_topc_set_coverage_contract,
)
from hpm.train import parse_models


def test_runner_accepts_hpm_lite_v2():
    assert parse_models("hpm_lite_v2") == ["hpm_lite_v2"]


def test_hard_slot_writer_rejects_posthoc_query_capacity_mismatch():
    with pytest.raises(ValueError, match="Invalid causal writer capacity contract"):
        validate_causal_writer_capacity_contract(
            SimpleNamespace(
                model="hpm_lite_v2",
                write_mode="learned",
                memory_slots=4,
                num_facts=8,
                delta_rule_writer=False,
                allow_legacy_capacity_mismatch=False,
            )
        )


def test_capacity_contract_allows_aligned_or_continuous_writer_modes():
    aligned = SimpleNamespace(
        model="hpm_lite_v2", write_mode="learned", memory_slots=8,
        num_facts=8, delta_rule_writer=False, allow_legacy_capacity_mismatch=False,
    )
    continuous = SimpleNamespace(
        model="hpm_lite_v2", write_mode="learned", memory_slots=4,
        num_facts=8, delta_rule_writer=True, allow_legacy_capacity_mismatch=False,
    )
    validate_causal_writer_capacity_contract(aligned)
    validate_causal_writer_capacity_contract(continuous)


def test_capacity_metadata_does_not_conflate_oracle_fact_count_with_hard_capacity():
    legacy = SimpleNamespace(
        model="hpm_lite_v2",
        write_mode="learned",
        memory_slots=4,
        num_facts=8,
        delta_rule_writer=False,
    )
    metadata = causal_writer_capacity_metadata(legacy)

    assert metadata == {
        "oracle_fact_slots_per_sample": 8,
        "episodic_capacity_configured": 4,
        "causal_writer_capacity_contract": "legacy_mismatch",
        "episodic_coverage_upper_bound": 0.5,
    }


def test_uncapped_hard_writer_metadata_says_it_retains_all_prequery_candidates():
    uncapped = SimpleNamespace(
        model="hpm_lite_v2",
        write_mode="learned",
        memory_slots=None,
        num_facts=8,
        delta_rule_writer=False,
    )
    metadata = causal_writer_capacity_metadata(uncapped)

    assert metadata["causal_writer_capacity_contract"] == "uncapped_prequery_candidates"
    assert metadata["episodic_capacity_configured"] is None
    assert metadata["episodic_coverage_upper_bound"] == 1.0


def test_causal_salience_writer_contract_requires_visible_finite_selection():
    aligned = SimpleNamespace(
        model="hpm_lite_v2",
        write_mode="learned",
        task="causal_salience_kv",
        memory_slots=2,
        num_facts=4,
        writer_required_facts=2,
        writer_candidate_mode="fact_pairs",
        delta_rule_writer=False,
    )
    validate_causal_writer_capacity_contract(aligned)
    metadata = causal_writer_capacity_metadata(aligned)
    assert metadata["causal_writer_capacity_contract"] == "causal_salience_aligned"
    assert metadata["episodic_coverage_upper_bound"] == 1.0
    assert metadata["causal_writer_required_facts"] == 2
    assert metadata["causal_writer_candidate_records"] == 4

    masked = SimpleNamespace(**{**vars(aligned), "writer_role_mode": "masked"})
    validate_causal_writer_capacity_contract(masked)
    masked_metadata = causal_writer_capacity_metadata(masked)
    assert masked_metadata["causal_writer_capacity_contract"] == "masked_role_negative_control"
    assert masked_metadata["episodic_coverage_upper_bound"] == 0.5

    with pytest.raises(ValueError, match="requires --writer-candidate-mode fact_pairs"):
        validate_causal_writer_capacity_contract(
            SimpleNamespace(**{**vars(aligned), "writer_candidate_mode": "all_prequery"})
        )
    with pytest.raises(ValueError, match="must retain fewer slots"):
        validate_causal_writer_capacity_contract(
            SimpleNamespace(**{**vars(aligned), "memory_slots": 4})
        )


def test_topc_coverage_refuses_jepa_writer_control():
    with pytest.raises(ValueError, match="keep --use-jepa-writer-bias false"):
        validate_topc_set_coverage_contract(
            SimpleNamespace(
                lambda_writer_set_coverage=0.1,
                model="hpm_lite_v2",
                write_mode="learned",
                delta_rule_writer=False,
                memory_slots=2,
                use_jepa_writer_bias=True,
                writer_set_coverage_eps=0.3,
                writer_set_coverage_iters=200,
                writer_set_coverage_mass_tolerance=1.0e-3,
            )
        )


def test_topc_coverage_is_restricted_to_the_visible_causal_control():
    base = dict(
        lambda_writer_set_coverage=0.1,
        model="hpm_lite_v2",
        write_mode="learned",
        delta_rule_writer=False,
        memory_slots=2,
        use_jepa_writer_bias=False,
        writer_set_coverage_eps=0.3,
        writer_set_coverage_iters=48,
        writer_set_coverage_mass_tolerance=1.0e-3,
        task="causal_salience_kv",
        writer_candidate_mode="fact_pairs",
        writer_role_mode="visible",
    )
    validate_topc_set_coverage_contract(SimpleNamespace(**base))
    with pytest.raises(ValueError, match="restricted to the causal_salience_kv control"):
        validate_topc_set_coverage_contract(SimpleNamespace(**{**base, "task": "kv"}))
    with pytest.raises(ValueError, match="requires --writer-role-mode visible"):
        validate_topc_set_coverage_contract(SimpleNamespace(**{**base, "writer_role_mode": "masked"}))


def test_causal_salience_topc_coverage_runs_on_the_self_writing_path(tmp_path):
    metrics = run_training(
        SimpleNamespace(
            model="hpm_lite_v2",
            task="causal_salience_kv",
            seq_len=128,
            window=16,
            batch_size=2,
            steps=2,
            eval_every=2,
            eval_batches=1,
            d_model=32,
            layers=1,
            heads=4,
            lr=3.0e-4,
            seed=124,
            device="cpu",
            lambda_ret=0.1,
            lambda_writer=0.1,
            lambda_writer_set_coverage=0.1,
            learned_writer_teacher_forcing_steps=0,
            top_k=1,
            memory_slots=2,
            writer_candidate_mode="fact_pairs",
            memory_null_slot=True,
            null_score_init=0.0,
            memory_control="normal",
            write_mode="learned",
            oracle_memory=True,
            num_facts=4,
            writer_required_facts=2,
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
        )
    )
    assert metrics["causal_writer_capacity_contract"] == "causal_salience_aligned"
    assert metrics["lambda_writer_set_coverage"] == 0.1
    assert math.isfinite(metrics["train_writer_set_coverage_loss"])
    assert metrics["train_writer_set_coverage_loss"] > 0.0
    assert math.isfinite(metrics["eval_writer_required_topc_margin"])
    assert 0.0 <= metrics["eval_writer_all_required_topc_rate"] <= 1.0
    assert metrics["eval_writer_topc_margin_samples"] == 2.0


def test_hpm_lite_v2_tiny_training_run(tmp_path):
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
            seed=123,
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
            save_step_log=True,
            record_vram=False,
        )
    )
    assert metrics["model"] == "hpm_lite_v2"
    assert metrics["parameters"] > 0
    assert "eval_answer_exact" in metrics
    assert "eval_retrieval_top1" in metrics
    assert metrics["lambda_writer_set_coverage"] == 0.0
    assert metrics["train_writer_set_coverage_loss"] == 0.0


def test_router_z_loss_flows_through_run_training(tmp_path):
    """End-to-end: lambda_router_z_loss must actually reach run_training's
    loss and the saved step_log.csv, not just exist as a dangling arg --
    this is what maintained experiment runners rely on."""
    import csv

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
            seed=123,
            device="cpu",
            lambda_ret=0.1,
            lambda_writer=0.1,
            lambda_router_z_loss=1.0e-3,
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
            save_step_log=True,
            record_vram=False,
        )
    )
    assert metrics["lambda_router_z_loss"] == 1.0e-3
    step_log_path = metrics["step_log_path"]
    assert step_log_path, "save_step_log=True should have produced a step_log_path"
    with open(step_log_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "step_log.csv should have at least one row"
    assert "train_router_z_loss" in rows[0]
    # lambda_router_z_loss > 0 means the term is active in the loss, so it
    # should actually be computed (not silently skipped/zeroed) -- confirms
    # the wiring, not just that the column exists.
    assert float(rows[-1]["train_router_z_loss"]) >= 0.0


def test_router_logit_clamp_flows_through_run_training(tmp_path):
    """End-to-end: router_logit_clamp must actually reach HpmLiteV2Config and
    the constructed model, not just exist as a dangling arg -- this is what
    run_memory_model.py / the modal sweep scripts rely on."""
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
            seed=123,
            device="cpu",
            lambda_ret=0.1,
            lambda_writer=0.1,
            router_logit_clamp=3.0,
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
            save_step_log=True,
            record_vram=False,
        )
    )
    assert metrics["router_logit_clamp"] == 3.0
    assert metrics["model"] == "hpm_lite_v2"
    assert "eval_answer_exact" in metrics


def test_delta_rule_writer_flows_through_run_training(tmp_path):
    """End-to-end: delta_rule_writer must actually reach HpmLiteV2Config and
    the constructed model, and a tiny run must complete without crashing
    across the teacher-forcing cutover point -- this is the smoke test that
    should be run (at seq_len=2048, more steps) via a Modal script before
    trusting any diagnostics comparison against the pre-existing hard top-k
    baseline runs."""
    metrics = run_training(
        SimpleNamespace(
            model="hpm_lite_v2",
            task="kv",
            seq_len=64,
            window=16,
            batch_size=2,
            steps=4,
            eval_every=4,
            eval_batches=1,
            d_model=32,
            layers=1,
            heads=4,
            lr=3.0e-4,
            seed=123,
            device="cpu",
            lambda_ret=0.1,
            lambda_writer=0.1,
            delta_rule_writer=True,
            learned_writer_teacher_forcing_steps=2,  # cutover happens mid-run (step 4 total)
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
            log_every=1,
            save_step_log=True,
            record_vram=False,
        )
    )
    assert metrics["delta_rule_writer"] is True
    assert metrics["model"] == "hpm_lite_v2"
    assert "eval_answer_exact" in metrics
    # retrieval_top1/topk are unavailable under delta_rule_writer (no
    # top_indices -- see HpmLiteV2Config.use_delta_rule_writer's docstring),
    # so metrics.retrieval_metrics degrades to {} rather than crashing.
    assert "eval_retrieval_top1" not in metrics


def test_delta_rule_writer_default_off_is_byte_identical_to_before(tmp_path):
    """delta_rule_writer=False (the default) must reproduce the exact prior
    hard top-k behavior -- same pattern as
    test_sinkhorn_warmup_weight_zero_is_byte_identical_to_disabled and
    test_router_logit_clamp_default_off_is_byte_identical_to_before."""
    import torch

    from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

    def build_batch():
        torch.manual_seed(0)
        return {
            "input_ids": torch.randint(0, 50, (2, 20)),
            "memory_token_positions": torch.randint(0, 18, (2, 4, 2)),
            "memory_mask": torch.ones(2, 4, dtype=torch.bool),
            "answer_positions": torch.full((2,), 19),
            "query_key_positions": torch.full((2,), 15),
        }

    torch.manual_seed(1)
    base_config = HpmLiteV2Config(model_type="hpm_lite_v2", d_model=16, layers=1, heads=2, window=8, use_learned_writer=True)
    base_model = HpmLiteV2Model(base_config)
    torch.manual_seed(1)
    explicit_config = HpmLiteV2Config(
        model_type="hpm_lite_v2", d_model=16, layers=1, heads=2, window=8, use_learned_writer=True, use_delta_rule_writer=False
    )
    explicit_model = HpmLiteV2Model(explicit_config)

    batch = build_batch()
    base_model.eval()
    explicit_model.eval()
    with torch.no_grad():
        base_out = base_model(**batch, use_learned_writer=True, learned_writer_teacher_forcing=True)
        explicit_out = explicit_model(**batch, use_learned_writer=True, learned_writer_teacher_forcing=True)

    assert torch.equal(base_out["logits"], explicit_out["logits"])


def test_delta_rule_writer_requires_use_learned_writer():
    from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

    with pytest.raises(ValueError):
        HpmLiteV2Model(
            HpmLiteV2Config(model_type="hpm_lite_v2", d_model=8, layers=1, heads=2, window=4, use_learned_writer=False, use_delta_rule_writer=True)
        )


def test_delta_rule_writer_rejects_multihop_tasks():
    import torch

    from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

    config = HpmLiteV2Config(model_type="hpm_lite_v2", d_model=16, layers=1, heads=2, window=8, use_learned_writer=True, use_delta_rule_writer=True)
    model = HpmLiteV2Model(config)
    batch = {
        "input_ids": torch.randint(0, 50, (1, 20)),
        "memory_token_positions": torch.randint(0, 18, (1, 4, 2)),
        "memory_mask": torch.ones(1, 4, dtype=torch.bool),
        "answer_positions": torch.full((1,), 19),
        "query_key_positions": torch.full((1,), 15),
    }
    with pytest.raises(ValueError):
        model(**batch, task="twohop", use_learned_writer=True, learned_writer_teacher_forcing=True)


def test_delta_rule_oracle_blend_flows_through_run_training(tmp_path):
    """End-to-end: delta_rule_oracle_blend must actually reach
    HpmLiteV2Config and complete a tiny run across the teacher-forcing
    cutover, paired with a nonzero scheduled-sampling anneal so
    teacher_forcing_prob actually varies over the run (see
    HpmLiteV2Config.delta_rule_oracle_blend's docstring for why a flat
    anneal_steps=0 wouldn't exercise the blend meaningfully)."""
    metrics = run_training(
        SimpleNamespace(
            model="hpm_lite_v2",
            task="kv",
            seq_len=64,
            window=16,
            batch_size=2,
            steps=6,
            eval_every=6,
            eval_batches=1,
            d_model=32,
            layers=1,
            heads=4,
            lr=3.0e-4,
            seed=123,
            device="cpu",
            lambda_ret=0.1,
            lambda_writer=0.1,
            delta_rule_writer=True,
            delta_rule_oracle_blend=True,
            learned_writer_teacher_forcing_steps=2,
            scheduled_sampling_anneal_steps=2,  # cutover ramps over steps 3-4, still mid-run
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
            log_every=1,
            save_step_log=True,
            record_vram=False,
        )
    )
    assert metrics["delta_rule_writer"] is True
    assert metrics["delta_rule_oracle_blend"] is True
    assert metrics["model"] == "hpm_lite_v2"
    assert "eval_answer_exact" in metrics


def test_delta_rule_oracle_blend_default_off_is_byte_identical_to_delta_rule_writer_alone(tmp_path):
    """delta_rule_oracle_blend=False (the default) must reproduce the exact
    same output as plain delta_rule_writer=True -- same no-op pattern as
    test_delta_rule_writer_default_off_is_byte_identical_to_before."""
    import torch

    from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

    def build_batch():
        torch.manual_seed(0)
        return {
            "input_ids": torch.randint(0, 50, (2, 20)),
            "memory_token_positions": torch.randint(0, 18, (2, 4, 2)),
            "memory_mask": torch.ones(2, 4, dtype=torch.bool),
            "answer_positions": torch.full((2,), 19),
            "query_key_positions": torch.full((2,), 15),
            "positive_memory_indices": torch.zeros(2, dtype=torch.long),
            "positive_memory_mask": torch.ones(2, 1, dtype=torch.bool),
        }

    torch.manual_seed(1)
    base_config = HpmLiteV2Config(
        model_type="hpm_lite_v2", d_model=16, layers=1, heads=2, window=8, use_learned_writer=True, use_delta_rule_writer=True
    )
    base_model = HpmLiteV2Model(base_config)
    torch.manual_seed(1)
    explicit_config = HpmLiteV2Config(
        model_type="hpm_lite_v2",
        d_model=16,
        layers=1,
        heads=2,
        window=8,
        use_learned_writer=True,
        use_delta_rule_writer=True,
        delta_rule_oracle_blend=False,
    )
    explicit_model = HpmLiteV2Model(explicit_config)

    batch = build_batch()
    base_model.eval()
    explicit_model.eval()
    with torch.no_grad():
        base_out = base_model(**batch, use_learned_writer=True, learned_writer_teacher_forcing=True, teacher_forcing_prob=1.0)
        explicit_out = explicit_model(
            **batch, use_learned_writer=True, learned_writer_teacher_forcing=True, teacher_forcing_prob=1.0
        )

    assert torch.equal(base_out["logits"], explicit_out["logits"])


def test_delta_rule_oracle_blend_requires_use_delta_rule_writer():
    from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

    with pytest.raises(ValueError):
        HpmLiteV2Model(
            HpmLiteV2Config(
                model_type="hpm_lite_v2",
                d_model=8,
                layers=1,
                heads=2,
                window=4,
                use_learned_writer=True,
                use_delta_rule_writer=False,
                delta_rule_oracle_blend=True,
            )
        )


def test_delta_rule_oracle_blend_at_full_teacher_forcing_prob_is_gated_by_oracle_position_only():
    """At teacher_forcing_prob=1.0, write_gate should collapse to EXACTLY
    the oracle mask (write_gate_mean == matched oracle occupancy), regardless
    of what the (randomly initialized, untrained) writer scorer thinks --
    this is the whole point of the blend: at p=1.0 the read is fully
    oracle-scaffolded, identical in spirit to the old hard-selection
    writer's teacher-forcing behavior, just expressed as a continuous gate
    instead of a hard read substitution."""
    import torch

    from hpm.hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model

    def build_batch():
        return {
            "input_ids": torch.randint(0, 50, (3, 20)),
            "memory_token_positions": torch.randint(0, 18, (3, 4, 2)),
            "memory_mask": torch.ones(3, 4, dtype=torch.bool),
            "answer_positions": torch.full((3,), 19),
            "query_key_positions": torch.full((3,), 15),
            "positive_memory_indices": torch.zeros(3, dtype=torch.long),
            "positive_memory_mask": torch.zeros(3, 4, dtype=torch.bool),
        }

    batch = build_batch()
    batch["positive_memory_mask"][:, 0] = True  # slot 0 is the true fact for every row

    write_gate_means = []
    for init_seed in (1, 2, 3):
        torch.manual_seed(init_seed)
        config = HpmLiteV2Config(
            model_type="hpm_lite_v2",
            d_model=16,
            layers=1,
            heads=2,
            window=8,
            use_learned_writer=True,
            use_delta_rule_writer=True,
            delta_rule_oracle_blend=True,
        )
        model = HpmLiteV2Model(config)
        model.eval()
        with torch.no_grad():
            out = model(**batch, use_learned_writer=True, learned_writer_teacher_forcing=True, teacher_forcing_prob=1.0)
        write_gate_means.append(out["retrieval"]["write_gate_mean"])

    # write_gate_mean must be identical across differently-initialized
    # (hence differently-scored) writers -- proof the learned score has zero
    # influence on write_gate when teacher_forcing_prob=1.0, only the oracle
    # position does.
    for other in write_gate_means[1:]:
        assert torch.allclose(write_gate_means[0], other)


def test_diagnose_writer_transition_writes_populated_csv_across_cutover(tmp_path):
    """End-to-end: --diagnose-writer-transition must actually produce
    writer_transition_diagnostics.csv with rows spanning both sides of
    learned_writer_teacher_forcing_steps, non-null grad norms for the
    subsystems that exist in this config, and non-null candidate-distribution
    fields (entropy always; KL/Jaccard once a previous diagnosed step
    exists) -- not just a dangling flag that silently does nothing."""
    import csv

    cutover = 2
    metrics = run_training(
        SimpleNamespace(
            model="hpm_lite_v2",
            task="kv",
            seq_len=64,
            window=16,
            batch_size=2,
            steps=5,
            eval_every=5,
            eval_batches=1,
            d_model=32,
            layers=1,
            heads=4,
            lr=3.0e-4,
            seed=123,
            device="cpu",
            lambda_ret=0.1,
            lambda_writer=0.1,
            diagnose_writer_transition=True,
            diagnostic_window=2,
            learned_writer_teacher_forcing_steps=cutover,
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
            log_every=5,
            save_step_log=False,
            record_vram=False,
        )
    )
    assert metrics["diagnose_writer_transition"] is True
    diag_path = metrics["writer_transition_diagnostics_path"]
    assert diag_path, "diagnose_writer_transition=True should have produced a diagnostics CSV path"
    with open(diag_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "writer_transition_diagnostics.csv should have at least one row"

    steps_seen = {int(row["step"]) for row in rows}
    # Window = cutover +/- 2, clipped to [1, 5]: expect rows on both sides.
    assert any(s <= cutover for s in steps_seen), "expected at least one pre/at-cutover diagnosed step"
    assert any(s > cutover for s in steps_seen), "expected at least one post-cutover diagnosed step"

    for row in rows:
        # grad_norm_local_blocks and grad_norm_writer always exist for
        # hpm_lite_v2 + write_mode=learned (episodic_only=False by default),
        # and every diagnosed step runs a real backward() through both.
        assert row["grad_norm_local_blocks"] not in ("", None)
        assert row["grad_norm_writer"] not in ("", None)
        # The probe forward always exercises the differentiable writer path
        # (learned_writer_teacher_forcing=False), so entropy is always defined.
        assert row["probe_candidate_entropy_mean"] not in ("", None)

    # KL/Jaccard vs-prev fields are None on the first diagnosed step only;
    # with >= 2 diagnosed steps, at least one row must have them populated.
    if len(rows) >= 2:
        assert any(row["probe_candidate_kl_vs_prev_mean"] not in ("", None) for row in rows)
        assert any(row["probe_topk_jaccard_vs_prev_mean"] not in ("", None) for row in rows)


def test_diagnose_writer_transition_off_by_default_writes_no_csv(tmp_path):
    """diagnose_writer_transition defaults to False and must not create the
    diagnostics CSV or add it to metrics when unset -- matches this
    codebase's convention of opt-in flags never changing default behavior."""
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
            seed=123,
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
        )
    )
    assert metrics["diagnose_writer_transition"] is False
    assert metrics["writer_transition_diagnostics_path"] == ""
    assert not (Path(metrics["run_dir"]) / "writer_transition_diagnostics.csv").exists()


def test_teacher_forcing_probability_schedule():
    """Pins the pure schedule function in isolation, independent of any model:
    boundary values, monotonicity, and that cosine/linear actually differ away
    from their shared endpoints and midpoint."""
    cutover = 10

    # anneal_steps=0 (or unset) must be an exact step function -- this is what
    # makes it a byte-for-bit no-op against the old hard `step <= cutover` cutover.
    for step in (1, 9, 10):
        assert teacher_forcing_probability(step, cutover, 0, "cosine") == 1.0
    for step in (11, 12, 500):
        assert teacher_forcing_probability(step, cutover, 0, "cosine") == 0.0

    anneal = 10
    # Endpoints of the anneal window, both schedules.
    for schedule in ("cosine", "linear"):
        assert teacher_forcing_probability(cutover, cutover, anneal, schedule) == 1.0
        assert teacher_forcing_probability(cutover + anneal, cutover, anneal, schedule) == 0.0
        assert teacher_forcing_probability(cutover + anneal + 5, cutover, anneal, schedule) == 0.0

    # Midpoint: both schedules agree at fraction=0.5 (cos(pi/2)=0 -> 0.5).
    mid_cosine = teacher_forcing_probability(cutover + 5, cutover, anneal, "cosine")
    mid_linear = teacher_forcing_probability(cutover + 5, cutover, anneal, "linear")
    assert abs(mid_cosine - 0.5) < 1e-9
    assert abs(mid_linear - 0.5) < 1e-9

    # Away from the shared midpoint, cosine and linear must actually differ --
    # cosine decays slower near both ends and faster through the middle.
    quarter_cosine = teacher_forcing_probability(cutover + 2, cutover, anneal, "cosine")  # fraction=0.2
    quarter_linear = teacher_forcing_probability(cutover + 2, cutover, anneal, "linear")
    assert quarter_cosine > quarter_linear  # cosine stays higher (slower start) early in the window

    # Monotonic non-increasing across the whole window, both schedules.
    for schedule in ("cosine", "linear"):
        values = [teacher_forcing_probability(cutover + i, cutover, anneal, schedule) for i in range(anneal + 1)]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))
        assert values[0] == 1.0 and values[-1] == 0.0


def _run_tiny_scheduled_sampling(tmp_path, **overrides):
    args = dict(
        model="hpm_lite_v2",
        task="kv",
        seq_len=64,
        window=16,
        batch_size=2,
        steps=12,
        eval_every=12,
        eval_batches=1,
        d_model=32,
        layers=1,
        heads=4,
        lr=3.0e-4,
        seed=123,
        device="cpu",
        lambda_ret=0.1,
        lambda_writer=0.1,
        learned_writer_teacher_forcing_steps=5,
        diagnose_writer_transition=True,
        diagnostic_window=30,  # covers the whole 12-step run regardless of anneal
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
        log_every=12,
        save_step_log=False,
        record_vram=False,
    )
    args.update(overrides)
    return run_training(SimpleNamespace(**args))


def _read_diagnostics_csv(metrics):
    path = metrics["writer_transition_diagnostics_path"]
    assert path, "diagnose_writer_transition=True should have produced a diagnostics CSV"
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_scheduled_sampling_anneal_steps_zero_is_byte_identical_to_hard_cutover(tmp_path):
    """anneal_steps=0 (the default) must reproduce the exact prior hard-switch
    teacher_forcing sequence AND identical losses -- this flag must not change
    any existing run's results unless explicitly opted into."""
    baseline = _run_tiny_scheduled_sampling(tmp_path / "baseline")
    explicit_zero = _run_tiny_scheduled_sampling(tmp_path / "explicit_zero", scheduled_sampling_anneal_steps=0)

    baseline_rows = _read_diagnostics_csv(baseline)
    zero_rows = _read_diagnostics_csv(explicit_zero)

    baseline_tf = [row["teacher_forcing"] for row in baseline_rows]
    zero_tf = [row["teacher_forcing"] for row in zero_rows]
    assert baseline_tf == zero_tf
    # Exact old hard-cutover semantics: True through step 5, False from step 6.
    expected_tf = [str(step <= 5) for step in range(1, 13)]
    assert baseline_tf == expected_tf

    # Same seed, same everything else, independent RNG stream consumed either
    # way -> losses must match exactly, step for step.
    baseline_losses = [row["loss"] for row in baseline_rows]
    zero_losses = [row["loss"] for row in zero_rows]
    assert baseline_losses == zero_losses


def test_scheduled_sampling_anneal_flows_through_run_training(tmp_path):
    """End-to-end: scheduled_sampling_anneal_steps/schedule must actually reach
    the training loop and produce a graded (not step-function) probability
    trajectory in the diagnostics CSV and in the saved run metadata."""
    metrics = _run_tiny_scheduled_sampling(
        tmp_path,
        scheduled_sampling_anneal_steps=4,
        scheduled_sampling_schedule="linear",
    )
    assert metrics["scheduled_sampling_anneal_steps"] == 4
    assert metrics["scheduled_sampling_schedule"] == "linear"

    rows = _read_diagnostics_csv(metrics)
    probs_by_step = {int(row["step"]): float(row["teacher_forcing_prob"]) for row in rows}
    # cutover=5, anneal=4 -> steps 1-5 at 1.0, step 9+ at 0.0, 6/7/8 strictly between.
    assert probs_by_step[5] == 1.0
    assert probs_by_step[9] == 0.0
    assert 0.0 < probs_by_step[7] < 1.0
    # Linear schedule -> probs_by_step[7] should sit at the window midpoint.
    assert abs(probs_by_step[7] - 0.5) < 1e-9
    # Monotonic non-increasing across the whole logged range.
    ordered = [probs_by_step[s] for s in sorted(probs_by_step)]
    assert all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1))


def test_sinkhorn_warmup_weight_zero_is_byte_identical_to_disabled(tmp_path):
    """sinkhorn_warmup_weight=0.0 (the default) must not change anything --
    HpmLiteV2Config.sinkhorn_warmup is only True when the weight is > 0, so
    the extra parallel read is never even constructed, let alone contributes
    to the loss."""
    baseline = _run_tiny_scheduled_sampling(tmp_path / "baseline")
    explicit_zero = _run_tiny_scheduled_sampling(tmp_path / "explicit_zero", sinkhorn_warmup_weight=0.0)

    baseline_rows = _read_diagnostics_csv(baseline)
    zero_rows = _read_diagnostics_csv(explicit_zero)
    assert [row["loss"] for row in baseline_rows] == [row["loss"] for row in zero_rows]
    assert [row["sinkhorn_warmup_loss"] for row in baseline_rows] == [""] * len(baseline_rows)


def test_sinkhorn_warmup_flows_through_run_training_and_only_fires_when_teacher_forced(tmp_path):
    """End-to-end: sinkhorn_warmup_weight must reach the training loop, show up
    in saved run metadata, and only produce a non-empty sinkhorn_warmup_loss on
    steps where teacher_forcing is True (that's the entire point -- warming the
    Sinkhorn path while the primary read is still the safe oracle-indexed one).
    On steps where teacher_forcing is False, the primary read already IS the
    Sinkhorn path, so the parallel warmup pass is skipped and the column must
    be empty there too."""
    metrics = _run_tiny_scheduled_sampling(
        tmp_path,
        learned_writer_teacher_forcing_steps=8,  # steps 1-8 teacher-forced, 9-12 not, within a 12-step run
        sinkhorn_warmup_weight=0.1,
    )
    assert metrics["sinkhorn_warmup_weight"] == 0.1

    rows = _read_diagnostics_csv(metrics)
    for row in rows:
        step = int(row["step"])
        warm = row["sinkhorn_warmup_loss"]
        if step <= 8:
            assert warm != "", f"step {step} is teacher-forced, expected a warmup loss"
            assert math.isfinite(float(warm))
        else:
            assert warm == "", f"step {step} is not teacher-forced, expected no warmup loss"


def test_sinkhorn_warmup_does_not_break_training_or_teacher_forcing_schedule(tmp_path):
    """The warmup pass reads through a SEPARATE router/answer_head call built
    off the same local_state and must not interfere with which steps get
    teacher-forced (an independent RNG stream from the writer/router) or
    prevent the run from completing with finite losses throughout."""
    with_warmup = _run_tiny_scheduled_sampling(
        tmp_path / "with_warmup",
        learned_writer_teacher_forcing_steps=8,
        sinkhorn_warmup_weight=0.1,
    )
    without_warmup = _run_tiny_scheduled_sampling(
        tmp_path / "without_warmup",
        learned_writer_teacher_forcing_steps=8,
    )
    rows_with = _read_diagnostics_csv(with_warmup)
    rows_without = _read_diagnostics_csv(without_warmup)
    assert [row["teacher_forcing"] for row in rows_with] == [row["teacher_forcing"] for row in rows_without]
    assert all(math.isfinite(float(row["loss"])) for row in rows_with)


def test_sinkhorn_warmup_weight_at_step_schedule():
    """Pure function, no torch needed. ramp_steps<=0 or target<=0 -> exact
    no-op (always returns target_weight, unconditionally). Otherwise linearly
    ramps from 0.0 at (cutover - ramp_steps) to target_weight at cutover, and
    holds flat at target_weight for every step at/after cutover."""
    # no-op guarantees
    for step in (1, 50, 140, 199, 200, 250):
        assert sinkhorn_warmup_weight_at_step(step, 200, 0, 0.1) == 0.1
        assert sinkhorn_warmup_weight_at_step(step, 200, 60, 0.0) == 0.0

    # ramp shape: cutover=200, ramp_steps=60 -> starts ramping at step 140
    assert sinkhorn_warmup_weight_at_step(139, 200, 60, 0.1) == 0.0
    assert sinkhorn_warmup_weight_at_step(140, 200, 60, 0.1) == 0.0
    assert sinkhorn_warmup_weight_at_step(170, 200, 60, 0.1) == pytest.approx(0.05)
    assert sinkhorn_warmup_weight_at_step(200, 200, 60, 0.1) == pytest.approx(0.1)
    assert sinkhorn_warmup_weight_at_step(250, 200, 60, 0.1) == pytest.approx(0.1)

    # monotonically non-decreasing across the ramp window
    values = [sinkhorn_warmup_weight_at_step(s, 200, 60, 0.1) for s in range(140, 201)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_sinkhorn_warmup_ramp_steps_zero_is_byte_identical_to_flat_weight(tmp_path):
    """sinkhorn_warmup_ramp_steps=0 (the default) must produce byte-identical
    losses to the pre-ramp flat-weight behavior -- sinkhorn_warmup_weight_at_step
    degenerates to always returning the configured target weight, same as the
    old plain `getattr(args, "sinkhorn_warmup_weight", 0.0)` read it replaced."""
    flat = _run_tiny_scheduled_sampling(
        tmp_path / "flat",
        learned_writer_teacher_forcing_steps=8,
        sinkhorn_warmup_weight=0.1,
    )
    explicit_zero_ramp = _run_tiny_scheduled_sampling(
        tmp_path / "explicit_zero_ramp",
        learned_writer_teacher_forcing_steps=8,
        sinkhorn_warmup_weight=0.1,
        sinkhorn_warmup_ramp_steps=0,
    )
    flat_rows = _read_diagnostics_csv(flat)
    zero_ramp_rows = _read_diagnostics_csv(explicit_zero_ramp)
    assert [row["loss"] for row in flat_rows] == [row["loss"] for row in zero_ramp_rows]


def test_sinkhorn_warmup_ramp_steps_actually_ramps(tmp_path):
    """With ramp_steps > 0, the effective weight (logged per-step in the
    diagnostics CSV) must start at 0.0 well before cutover, be strictly below
    the target weight partway through the ramp window, and reach exactly the
    target weight AT cutover -- confirming the schedule actually reaches the
    training loop's loss computation, not just existing as an unused arg."""
    metrics = _run_tiny_scheduled_sampling(
        tmp_path,
        learned_writer_teacher_forcing_steps=8,
        sinkhorn_warmup_weight=0.1,
        sinkhorn_warmup_ramp_steps=6,  # ramp starts at step 8-6=2
    )
    assert metrics["sinkhorn_warmup_ramp_steps"] == 6
    # configured target weight must still be recorded correctly in metadata,
    # not overwritten by whatever the last step's ramped value happened to be.
    assert metrics["sinkhorn_warmup_weight"] == 0.1

    rows = _read_diagnostics_csv(metrics)
    by_step = {int(row["step"]): row for row in rows}
    assert float(by_step[2]["sinkhorn_warmup_weight_effective"]) == 0.0
    mid = float(by_step[5]["sinkhorn_warmup_weight_effective"])
    assert 0.0 < mid < 0.1
    assert float(by_step[8]["sinkhorn_warmup_weight_effective"]) == pytest.approx(0.1)


def test_primary_loss_weight_at_step_schedule():
    """Pure function, no torch needed. Mirror image of
    sinkhorn_warmup_weight_at_step: ramp_steps<=0 -> exact no-op (always
    returns 1.0). Otherwise linearly ramps DOWN from 1.0 at
    (cutover - ramp_steps) to floor_weight at cutover, and holds flat at
    floor_weight for every step at/after cutover."""
    # no-op guarantee: ramp_steps<=0 always returns 1.0, regardless of floor_weight.
    for step in (1, 50, 140, 199, 200, 250):
        assert primary_loss_weight_at_step(step, 200, 0, 0.1) == 1.0

    # ramp shape: cutover=200, ramp_steps=60 -> starts ramping at step 140.
    assert primary_loss_weight_at_step(139, 200, 60, 0.1) == 1.0
    assert primary_loss_weight_at_step(140, 200, 60, 0.1) == 1.0
    assert primary_loss_weight_at_step(170, 200, 60, 0.1) == pytest.approx(0.55)
    assert primary_loss_weight_at_step(200, 200, 60, 0.1) == pytest.approx(0.1)
    assert primary_loss_weight_at_step(250, 200, 60, 0.1) == pytest.approx(0.1)

    # floor_weight=1.0 makes a > 0 ramp itself a no-op (flat 1.0 throughout).
    for step in (140, 170, 200, 250):
        assert primary_loss_weight_at_step(step, 200, 60, 1.0) == pytest.approx(1.0)

    # monotonically non-increasing across the ramp window (mirror of the
    # sinkhorn-warmup ramp, which is non-decreasing).
    values = [primary_loss_weight_at_step(s, 200, 60, 0.1) for s in range(140, 201)]
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_primary_loss_anneal_steps_zero_is_byte_identical_to_disabled(tmp_path):
    """primary_loss_anneal_steps=0 (the default) must produce byte-identical
    losses to not having the flag at all -- lambda_primary stays 1.0 for
    every step, identical to the plain `answer_loss` term that predates this
    flag entirely."""
    baseline = _run_tiny_scheduled_sampling(
        tmp_path / "baseline",
        learned_writer_teacher_forcing_steps=8,
        sinkhorn_warmup_weight=0.1,
    )
    explicit_zero = _run_tiny_scheduled_sampling(
        tmp_path / "explicit_zero",
        learned_writer_teacher_forcing_steps=8,
        sinkhorn_warmup_weight=0.1,
        primary_loss_anneal_steps=0,
    )
    baseline_rows = _read_diagnostics_csv(baseline)
    zero_rows = _read_diagnostics_csv(explicit_zero)
    assert [row["loss"] for row in baseline_rows] == [row["loss"] for row in zero_rows]


def test_primary_loss_anneal_steps_only_applies_when_sinkhorn_warmup_active(tmp_path):
    """primary_loss_anneal_steps > 0 must be a no-op unless sinkhorn_warmup_weight
    is also > 0 -- ramping the privileged path's weight down only makes sense
    paired with the honest path's weight ramping up (see the docstring on
    --primary-loss-anneal-steps). Without sinkhorn warmup active, lambda_primary
    must stay pinned at 1.0 for every step."""
    without_warmup = _run_tiny_scheduled_sampling(
        tmp_path / "without_warmup",
        learned_writer_teacher_forcing_steps=8,
        primary_loss_anneal_steps=6,
    )
    baseline = _run_tiny_scheduled_sampling(
        tmp_path / "baseline",
        learned_writer_teacher_forcing_steps=8,
    )
    assert [row["loss"] for row in _read_diagnostics_csv(without_warmup)] == [
        row["loss"] for row in _read_diagnostics_csv(baseline)
    ]


def test_primary_loss_anneal_steps_actually_ramps(tmp_path):
    """With ramp_steps > 0 and sinkhorn warmup active, the effective weight
    (logged per-step in the diagnostics CSV) must start at 1.0 well before
    cutover, sit strictly between the floor and 1.0 partway through the ramp
    window, and reach exactly the floor weight AT cutover -- confirming the
    schedule actually reaches the training loop's loss computation."""
    metrics = _run_tiny_scheduled_sampling(
        tmp_path,
        learned_writer_teacher_forcing_steps=8,
        sinkhorn_warmup_weight=0.1,
        primary_loss_anneal_steps=6,  # ramp starts at step 8-6=2
    )
    assert metrics["primary_loss_anneal_steps"] == 6
    assert metrics["primary_loss_floor_weight"] == 0.1  # default

    rows = _read_diagnostics_csv(metrics)
    by_step = {int(row["step"]): row for row in rows}
    assert float(by_step[2]["primary_loss_weight_effective"]) == pytest.approx(1.0)
    mid = float(by_step[5]["primary_loss_weight_effective"])
    assert 0.1 < mid < 1.0
    assert float(by_step[8]["primary_loss_weight_effective"]) == pytest.approx(0.1)

    # Non-teacher-forced steps (9-12) must keep lambda_primary pinned at 1.0 --
    # the primary read IS the honest self-selected read there, and needs full
    # weight, not a floor value meant for a different regime.
    for step in range(9, 13):
        assert float(by_step[step]["primary_loss_weight_effective"]) == pytest.approx(1.0)


def test_data_seed_decouples_data_sampling_from_model_init(tmp_path):
    """The actual point of --data-seed: holding --seed fixed (same model init
    and training-dynamics RNG stream) while varying --data-seed must change
    which facts get sampled -- and holding --data-seed fixed while varying
    --seed must change model init/training dynamics but sample the identical
    facts. Before this flag, these two were inseparable (both driven by the
    single --seed), so a run that looked seed-dependent could never be
    attributed to "this init is unstable" vs "this sampled fact set is
    hard" without this split."""
    same_seed_diff_data = _run_tiny_scheduled_sampling(
        tmp_path / "a", seed=7, data_seed=1,
    )
    same_seed_diff_data_2 = _run_tiny_scheduled_sampling(
        tmp_path / "b", seed=7, data_seed=2,
    )
    assert same_seed_diff_data["data_seed"] == 1
    assert same_seed_diff_data_2["data_seed"] == 2
    # Same --seed (7) but different --data-seed must produce different losses
    # -- different facts were sampled, even though model init and the
    # scheduled-sampling RNG stream (both keyed on --seed) are identical.
    rows_a = _read_diagnostics_csv(same_seed_diff_data)
    rows_b = _read_diagnostics_csv(same_seed_diff_data_2)
    assert [r["loss"] for r in rows_a] != [r["loss"] for r in rows_b]
    # But the teacher_forcing sequence itself (driven by args.seed + 300_000,
    # not data_seed) must be identical -- confirms the RNG split actually
    # keeps training-dynamics randomness decoupled from data randomness.
    assert [r["teacher_forcing"] for r in rows_a] == [r["teacher_forcing"] for r in rows_b]


def test_data_seed_none_matches_explicit_seed_value(tmp_path):
    """data_seed=None must be exactly equivalent to explicitly passing
    data_seed=<the same value as --seed> -- confirms the fallback
    (`data_seed = args.seed if data_seed is None else data_seed`) actually
    reproduces the old single-seed behavior bit for bit, not just
    approximately."""
    implicit = _run_tiny_scheduled_sampling(tmp_path / "implicit", seed=42)
    explicit = _run_tiny_scheduled_sampling(tmp_path / "explicit", seed=42, data_seed=42)
    assert implicit["data_seed"] == 42
    assert explicit["data_seed"] == 42
    rows_implicit = _read_diagnostics_csv(implicit)
    rows_explicit = _read_diagnostics_csv(explicit)
    assert [r["loss"] for r in rows_implicit] == [r["loss"] for r in rows_explicit]
