"""spatialml.features - leakage-safe spatial neighbourhood features.

Every feature depends only on cells within `cap_um` of the focal cell (self excluded). Nearest-type distances are
CAPPED at cap_um: an unbounded distance would encode cells far outside the feature radius, including the target ring.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .annotation import COARSE_TYPES
from .neighborhood import nearest_type_distance, ring_stats, shannon_entropy

UNASSIGNED = "Unassigned"


def type_codes(cell_type, types=COARSE_TYPES):
    all_types = list(types) + [UNASSIGNED]
    lut = {t: i for i, t in enumerate(all_types)}
    codes = pd.Series(np.asarray(cell_type, dtype=object)).map(lut).fillna(len(all_types) - 1).astype(int).to_numpy()
    return codes, all_types


def _rname(r):
    return int(r) if float(r).is_integer() else r


def build_spatial_features(xy, cell_type, radii=(15, 30), cap_um=30, knn_ks=(5, 10), types=COARSE_TYPES, chunk=20000):
    """Return a DataFrame of strict spatial features (one row per cell, in input order)."""
    xy = np.asarray(xy, dtype=np.float64)
    if max(radii) > cap_um:
        raise ValueError("radii must not exceed cap_um")
    codes, all_types = type_codes(cell_type, types)
    onehot = np.eye(len(all_types), dtype=np.float32)[codes]
    stats = ring_stats(xy, onehot, [(0, r) for r in radii], chunk=chunk)
    feat = {}
    for r in radii:
        st = stats[(0, r)]; n = st["n"].astype(np.float64); cnt = st["counts"]; rn = _rname(r)
        feat[f"n_cells_r{rn}"] = n
        feat[f"density_r{rn}"] = n / (np.pi * (r / 1000.0) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = cnt / n[:, None]
        frac[n == 0] = np.nan
        for k, t in enumerate(all_types):
            feat[f"frac_{t}_r{rn}"] = frac[:, k]
        asg = cnt[:, :-1]; na = asg.sum(1)
        feat[f"entropy_r{rn}"] = shannon_entropy(asg)
        with np.errstate(divide="ignore", invalid="ignore"):
            dom = asg.max(1) / na
            edge = np.hypot(st["vx"], st["vy"]) / n
            mnd = st["dsum"] / n
        dom[na == 0] = np.nan; edge[n == 0] = np.nan; mnd[n == 0] = np.nan
        feat[f"dominant_frac_r{rn}"] = dom
        feat[f"edge_asymmetry_r{rn}"] = edge
        feat[f"mean_nbr_dist_r{rn}"] = mnd
    tree = cKDTree(xy)
    kmax = max(knn_ks)
    dk, _ = tree.query(xy, k=kmax + 1, workers=-1)
    dk = np.minimum(dk.reshape(len(xy), -1)[:, 1:], cap_um)          # cap EACH neighbour distance (missing -> inf -> cap)
    for k in knn_ks:
        feat[f"knn_dist_cap{_rname(cap_um)}_k{k}"] = dk[:, :k].mean(1)
    for i, t in enumerate(types):
        d = nearest_type_distance(xy, codes, i)
        feat[f"dist_nearest_{t}_cap{_rname(cap_um)}"] = np.where(np.isnan(d), cap_um, np.minimum(d, cap_um))
    return pd.DataFrame(feat)


def reach_test(xy, cell_type, n_cells=30, seed=0, **kw):
    """Perturbation test of the reach guarantee: delete every cell farther than cap_um from a focal cell and recompute
    its features. Returns the largest absolute difference (should be ~0)."""
    xy = np.asarray(xy, dtype=np.float64); ct = np.asarray(cell_type, dtype=object)
    cap = kw.get("cap_um", 30)
    full = build_spatial_features(xy, ct, **kw)
    tree = cKDTree(xy); rng = np.random.default_rng(seed)
    worst = 0.0
    for i in rng.choice(len(xy), min(n_cells, len(xy)), replace=False):
        idx = np.array(sorted(tree.query_ball_point(xy[i], cap + 1e-9)))
        sub = build_spatial_features(xy[idx], ct[idx], **kw)
        a = full.iloc[i].to_numpy(float); b = sub.iloc[int(np.where(idx == i)[0][0])].to_numpy(float)
        if np.any(np.isnan(a) != np.isnan(b)):
            return float("inf")
        ok = ~np.isnan(a)
        if ok.any():
            worst = max(worst, float(np.abs(a[ok] - b[ok]).max()))
    return worst
