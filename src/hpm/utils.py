from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

# Must be set before CUDA initializes -- torch.use_deterministic_algorithms
# raises a RuntimeError on the first cuBLAS matmul without this, separately
# from warn_only (which only covers ops with no deterministic impl at all;
# cuBLAS has one, it just needs this to use it). Only overrides if unset, so
# a caller's own value (e.g. set in the Modal entrypoint) still wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Catches non-cudnn sources too (e.g. index_add_/scatter-add in embedding
    # backward), which is the more likely culprit for a memory/router model
    # like this one. warn_only=True so an op without a deterministic CUDA
    # kernel logs a warning instead of hard-crashing the run.
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value}")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)