"""spatialml.config - default configuration, loading, validation and seeding."""
import copy
import json
import random
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = {
    "seed": 42,
    "coordinate_units": "micron",
    "features": {"radii_um": [15, 30], "cap_um": 30, "knn_ks": [5, 10]},
    "target": {"annulus_um": [50, 100], "min_cells_in_annulus": 10, "k": 3},
    "cv": {"block_um": 500, "buffer_um": 100, "folds": 5},
    "model": {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 200,
              "l2_regularization": 1.0, "min_samples_leaf": 100},
    "evaluation": {"n_bootstrap": 1000, "max_train_cells": 80000},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def validate_config(cfg):
    """The leakage rules of the project, enforced in code."""
    f, t, cv = cfg["features"], cfg["target"], cfg["cv"]
    if max(f["radii_um"]) > f["cap_um"]:
        raise ValueError("feature radii must not exceed the reach cap")
    if f["cap_um"] >= t["annulus_um"][0]:
        raise ValueError("feature reach cap must be smaller than the inner edge of the target ring")
    if cv["buffer_um"] < t["annulus_um"][1]:
        raise ValueError("CV buffer must be at least the outer radius of the target ring")
    return cfg


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        if not Path(path).exists():
            raise FileNotFoundError(f"configuration file not found: {path}")
        cfg = _merge(cfg, json.load(open(path)))
    return validate_config(_merge(cfg, overrides))


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(path, "w"), indent=2)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    return np.random.default_rng(seed)
