"""Config loading and small runtime helpers."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = "configs/default.yaml"


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("AGRISIGHT_CONFIG", DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Environment overrides (handy on Colab: keep outputs on Google Drive).
    for env, key in (("AGRISIGHT_RUNS_DIR", "runs_dir"), ("AGRISIGHT_INDEX_DIR", "index_dir"),
                     ("AGRISIGHT_MANIFEST_DIR", "manifest_dir")):
        if os.environ.get(env):
            cfg["paths"][key] = os.environ[env]
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str | None = None):
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def manifest_path(cfg: dict, name: str) -> Path:
    """Resolve a manifest name like 'lab_train' to data/manifests/lab_train.csv."""
    p = Path(name)
    if p.suffix == ".csv" or p.exists():
        return p
    return Path(cfg["paths"]["manifest_dir"]) / f"{name}.csv"
