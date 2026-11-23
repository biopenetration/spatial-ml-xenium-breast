"""spatialml.neighborhood - leakage-safe spatial neighborhood utilities (Module 3).

All functions work on micron coordinates. Self is always excluded from neighborhoods.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def ring_stats(xy, onehot, rings, chunk=20000, tree=None):
    """Neighborhood statistics for several rings in a single pass.

    xy      : (n, 2) coordinates (um)
    onehot  : (n, T) one-hot cell-type matrix
    rings   : list of (r_in, r_out); a neighbor j of cell i is used if r_in <= d(i, j) <= r_out and j != i
    returns : {ring: dict(counts (n,T), n (n,), vx, vy, dsum)}
              vx, vy = sum of unit vectors cell->neighbor (for edge asymmetry), dsum = sum of distances
    """
    xy = np.asarray(xy, dtype=np.float64)
    n, T = onehot.shape
    tree = tree if tree is not None else cKDTree(xy)
    r_max = max(r for _, r in rings)
    out = {ring: dict(counts=np.zeros((n, T), np.float32), n=np.zeros(n, np.int32),
                      vx=np.zeros(n, np.float32), vy=np.zeros(n, np.float32),
                      dsum=np.zeros(n, np.float32)) for ring in rings}
    onehot = onehot.astype(np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        qt = cKDTree(xy[s:e])
        res = qt.sparse_distance_matrix(tree, r_max, output_type="ndarray")
        i = res["i"].astype(np.int64); j = res["j"].astype(np.int64); d = res["v"]
        gi = i + s
        keep = gi != j                      # exclude self
        i, j, d, gi = i[keep], j[keep], d[keep], gi[keep]
        safe = np.where(d > 0, d, 1.0)
        ux = np.where(d > 0, (xy[j, 0] - xy[gi, 0]) / safe, 0.0)
        uy = np.where(d > 0, (xy[j, 1] - xy[gi, 1]) / safe, 0.0)
        for ring in rings:
            r_in, r_out = ring
            m = (d >= r_in) & (d <= r_out)
            ii, jj = i[m], j[m]
            W = sp.csr_matrix((np.ones(m.sum(), np.float32), (ii, jj)), shape=(e - s, n))
            o = out[ring]
            o["counts"][s:e] = W @ onehot
            o["n"][s:e] = np.bincount(ii, minlength=e - s)
            o["vx"][s:e] = np.bincount(ii, weights=ux[m], minlength=e - s)
            o["vy"][s:e] = np.bincount(ii, weights=uy[m], minlength=e - s)
            o["dsum"][s:e] = np.bincount(ii, weights=d[m], minlength=e - s)
    return out


def shannon_entropy(counts):
    """Shannon entropy (nats) of count rows; NaN where the row sum is 0."""
    tot = counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / tot
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    h = np.maximum(h, 0.0)
    h[tot.ravel() == 0] = np.nan
    return h


def nearest_type_distance(xy, type_codes, type_index):
    """Distance from each cell to the nearest OTHER cell of the given type (NaN if there is no such cell)."""
    mask = type_codes == type_index
    n = int(mask.sum())
    if n == 0:
        return np.full(len(xy), np.nan)
    t = cKDTree(xy[mask])
    if n == 1:                                   # a single cell of this type: every other cell can still measure to it
        d, _ = t.query(xy, k=1, workers=-1)
        return np.where(mask, np.nan, d)
    d, _ = t.query(xy, k=2, workers=-1)
    return np.where(mask, d[:, 1], d[:, 0])


def make_spatial_folds(xy, block_um, n_folds, seed=0):
    """Assign square spatial blocks to folds, balancing cell counts greedily."""
    bx = np.floor((xy[:, 0] - xy[:, 0].min()) / block_um).astype(np.int64)
    by = np.floor((xy[:, 1] - xy[:, 1].min()) / block_um).astype(np.int64)
    block = bx * (by.max() + 1) + by
    ub, inv, cnt = np.unique(block, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ub))
    order = order[np.argsort(-cnt[order], kind="stable")]
    load = np.zeros(n_folds)
    fold_of_block = np.zeros(len(ub), np.int64)
    for b in order:
        f = int(np.argmin(load)); fold_of_block[b] = f; load[f] += cnt[b]
    return block, fold_of_block[inv]


def buffered_train_mask(xy, fold, test_fold, buffer_um):
    """Train = not in test fold AND at least buffer_um away from every test cell."""
    test = fold == test_fold
    d, _ = cKDTree(xy[test]).query(xy, k=1, workers=-1)
    train = (~test) & (d >= buffer_um)
    return train, test
