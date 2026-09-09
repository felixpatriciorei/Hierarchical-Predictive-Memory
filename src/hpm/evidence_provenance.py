"""Small provenance helpers for immutable HPM experiment evidence.

These helpers are intentionally stdlib-only so both local analysis scripts and
Modal local entrypoints can use them without adding dependencies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def evidence_set_sha256(seed_to_sha256: Mapping[int, str]) -> str:
    """Hash a canonical seed->artifact-hash mapping.

    File names and timestamps are deliberately excluded. The digest identifies
    the evidence content assigned to each seed, independent of where it lives.
    """
    payload = {str(int(seed)): str(digest) for seed, digest in sorted(seed_to_sha256.items())}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(canonical)


def find_collisions(output_dir: Path, seeds: Iterable[int]) -> list[Path]:
    output_dir = Path(output_dir)
    collisions: list[Path] = []
    for seed in seeds:
        for name in (
            f"seed{int(seed)}_composite_step_log.csv",
            f"seed{int(seed)}_router_dependency.json",
        ):
            p = output_dir / name
            if p.exists():
                collisions.append(p)
    return collisions


def require_collision_free(output_dir: Path, seeds: Iterable[int], *, allow_overwrite: bool = False) -> None:
    """Fail before an experiment if its canonical destinations already exist."""
    collisions = find_collisions(output_dir, seeds)
    if collisions and not allow_overwrite:
        rendered = "\n  ".join(str(p) for p in collisions)
        raise FileExistsError(
            "refusing to overwrite existing experiment evidence:\n  "
            + rendered
            + "\nUse a fresh --out-subdir for a new run. Pass --allow-overwrite only "
              "when destruction of prior evidence is intentional."
        )
