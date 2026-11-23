"""spatialml.pipeline - an end-to-end, leakage-safe run on synthetic data (demo and integration test)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import modeling as ml
from . import neighborhood as nbh
from .annotation import COARSE_TYPES
from .config import load_config, set_seed
from .demo import make_synthetic_tissue
from .features import build_spatial_features
from .targets import annulus_composition, class_margin, fit_target_classes, name_classes


def _bal_acc_ci(y, proba, idx, blk, n_boot, rng):
    codes, uni = pd.factorize(blk[idx]); nb = len(uni); K = proba.shape[1]
    Cb = ml.block_confusions(y[idx], np.nan_to_num(proba[idx]).argmax(1), codes, nb, K)
    point = float(ml.metrics_from_counts(Cb.sum(0))["balanced_accuracy"])
    boot = ml.metrics_from_counts(np.einsum("rb,bkl->rkl", ml.bootstrap_weights(nb, n_boot, rng), Cb))["balanced_accuracy"]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi), boot


def run_demo(out_dir, n_cells=6000, seed=0, n_bootstrap=200, block_um=300, config=None):
    """Synthetic tissue -> strict features -> target -> spatial CV -> results. Returns a dict and writes JSON/CSV."""
    cfg = load_config(overrides={"cv": {"block_um": block_um, "folds": 5}, "seed": seed, **(config or {})})
    rng = set_seed(cfg["seed"]); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    tissue = make_synthetic_tissue(n_cells, seed=seed)
    xy = tissue[["x", "y"]].to_numpy(); ct = tissue["cell_type"].to_numpy()
    fc, tc = cfg["features"], cfg["target"]
    feats = build_spatial_features(xy, ct, tuple(fc["radii_um"]), fc["cap_um"], tuple(fc["knn_ks"])).astype(np.float32)
    frac, n_ann, valid = annulus_composition(xy, ct, tuple(tc["annulus_um"]), min_cells=tc["min_cells_in_annulus"])
    labels = fit_target_classes(frac, valid, k=tc["k"], seed=cfg["seed"])
    names = name_classes(frac, labels); margin = class_margin(frac, labels)
    K = len(names); y = labels.copy()

    _, fold = nbh.make_spatial_folds(xy, cfg["cv"]["block_um"], cfg["cv"]["folds"], seed=cfg["seed"])
    blk = pd.factorize(np.floor(xy[:, 0] / cfg["cv"]["block_um"]) * 10000 + np.floor(xy[:, 1] / cfg["cv"]["block_um"]))[0]
    type_cols = [f"type_{t}" for t in COARSE_TYPES + ["Unassigned"]]
    X = feats.copy()
    for t in COARSE_TYPES + ["Unassigned"]:
        X[f"type_{t}"] = (ct == t).astype(np.float32)
    X["x"] = xy[:, 0].astype(np.float32); X["y"] = xy[:, 1].astype(np.float32)
    strict_cols = list(feats.columns)

    spl = ev.spatial_splits(xy, valid, fold, cfg["cv"]["buffer_um"], cfg["evaluation"]["max_train_cells"], cfg["seed"])
    rspl = ev.random_splits(valid, cfg["cv"]["folds"], cfg["evaluation"]["max_train_cells"], cfg["seed"])
    idx = np.where(valid)[0]
    P = {}
    P["own_type_baseline"] = ev.run_cv(X, y, spl, type_cols, "lookup", {"type_cols": type_cols}, cfg["seed"], K)
    P["spatial_strict"] = ev.run_cv(X, y, spl, strict_cols, "hgb", cfg["model"], cfg["seed"], K)
    P["coordinates_spatial_cv"] = ev.run_cv(X, y, spl, ["x", "y"], "hgb", cfg["model"], cfg["seed"], K)
    P["coordinates_random_cv"] = ev.run_cv(X, y, rspl, ["x", "y"], "hgb", cfg["model"], cfg["seed"], K)

    rows = []
    for name, p in P.items():
        pt, lo, hi, _ = _bal_acc_ci(y, p, idx, blk, n_bootstrap, rng)
        rows.append({"experiment": name, "balanced_accuracy": pt, "ci_low": lo, "ci_high": hi})
    res = pd.DataFrame(rows); res.to_csv(out / "demo_results.csv", index=False)
    summary = {"n_cells": int(n_cells), "n_valid": int(valid.sum()), "classes": [names[c] for c in sorted(names)],
               "mean_margin": float(np.nanmean(margin)), "chance": 1.0 / K,
               "results": {r["experiment"]: r["balanced_accuracy"] for r in rows}, "config": cfg}
    json.dump(summary, open(out / "demo_summary.json", "w"), indent=2)
    return summary
