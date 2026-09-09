from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_router_interactions.py"
    spec = importlib.util.spec_from_file_location("analyze_router_interactions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_numerical_roundoff_is_classified_as_zero(tmp_path: Path):
    m = _load_script_module()
    # Construct a mathematically zero interaction whose floating arithmetic can
    # land at tiny non-zero values. The analyzer should canonicalize it to zero.
    art = {
        "diagnostic_kind": m.EXPECTED_KIND,
        "seed": 0,
        "path_names": ["a", "b"],
        "condition_exact": {
            "full": {"x": 0.3},
            "drop_a": {"x": 0.2},
            "drop_b": {"x": 0.1},
            "drop_a__b": {"x": 0.0},
        },
        "_source_sha256": "0" * 64,
    }
    report = m.analyze([art])
    row = report["subtasks"]["x"]["a+b"]
    assert row["mean"] == 0.0
    assert row["positive_seed_count"] == 0
    assert row["negative_seed_count"] == 0
    assert row["zero_seed_count"] == 1
