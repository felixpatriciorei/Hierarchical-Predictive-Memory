from __future__ import annotations

from pathlib import Path

import pytest

from hpm.evidence_provenance import (
    evidence_set_sha256,
    find_collisions,
    require_collision_free,
    sha256_text,
)


def test_evidence_set_digest_is_order_independent_and_seed_sensitive():
    a = evidence_set_sha256({0: "aaa", 1: "bbb"})
    b = evidence_set_sha256({1: "bbb", 0: "aaa"})
    c = evidence_set_sha256({0: "bbb", 1: "aaa"})
    assert a == b
    assert a != c
    assert len(a) == 64


def test_sha256_text_is_content_addressed():
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_collision_guard_refuses_existing_seed_evidence(tmp_path: Path):
    existing = tmp_path / "seed0_router_dependency.json"
    existing.write_text("{}", encoding="utf-8")
    assert find_collisions(tmp_path, [0]) == [existing]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        require_collision_free(tmp_path, [0])
    require_collision_free(tmp_path, [0], allow_overwrite=True)


def test_collision_guard_allows_fresh_subdirectory(tmp_path: Path):
    (tmp_path / "seed0_router_dependency.json").write_text("{}", encoding="utf-8")
    fresh = tmp_path / "n0_repeat"
    require_collision_free(fresh, [0])
