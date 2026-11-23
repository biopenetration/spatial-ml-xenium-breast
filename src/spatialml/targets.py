"""spatialml.targets - microenvironment target from the composition of an annulus around each cell."""
import numpy as np
from scipy.spatial import cKDTree  # noqa: F401  (kept for API symmetry)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .annotation import COARSE_TYPES
from .features import type_codes
from .neighborhood import ring_stats


def annulus_composition(xy, cell_type, annulus=(50, 100), types=COARSE_TYPES, min_cells=10, chunk=20000):
    """Composition (fractions of assigned cells) in the ring r_in <= d <= r_out, self excluded.
    Returns (frac (n x T) with NaN where invalid, n_cells_in_ring, valid mask)."""
    codes, all_types = type_codes(cell_type, types)
    onehot = np.eye(len(all_types), dtype=np.float32)[codes]
    ring = (annulus[0], annulus[1])
    st = ring_stats(np.asarray(xy, float), onehot, [ring], chunk=chunk)[ring]
    cnt = st["counts"]; n_ann = st["n"]; n_asg = cnt[:, :-1].sum(1)
    valid = (n_ann >= min_cells) & (n_asg >= min_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = cnt[:, :-1] / n_asg[:, None]
    frac[~valid] = np.nan
    return frac, n_ann, valid


def hellinger(frac):
    return np.sqrt(np.nan_to_num(frac))


def fit_target_classes(frac, valid, k=3, seed=42, n_init=10):
    """k-means on Hellinger-transformed composition. Returns raw labels (-1 where invalid)."""
    H = hellinger(frac); vidx = np.where(valid)[0]
    km = KMeans(k, n_init=n_init, random_state=seed).fit(H[vidx])
    labels = np.full(len(frac), -1); labels[vidx] = km.labels_
    return labels


def name_classes(frac, labels, types=COARSE_TYPES):
    """Name classes from their mean composition; numbered by size (ME1 = largest)."""
    ks = sorted(int(c) for c in np.unique(labels) if c >= 0)
    means = {c: np.nanmean(frac[labels == c], axis=0) for c in ks}
    sizes = {c: int((labels == c).sum()) for c in ks}
    names = {}
    for rank, c in enumerate(sorted(ks, key=lambda c: -sizes[c]), start=1):
        o = np.argsort(-means[c]); t1, t2 = types[o[0]], types[o[1]]
        names[c] = f"ME{rank}_{t1}-rich" if means[c][o[0]] >= 0.55 else f"ME{rank}_{t1}/{t2}"
    return names


def class_margin(frac, labels):
    """Relative distance to the second-closest class centroid (0 = on the boundary, 1 = unambiguous)."""
    H = hellinger(frac); valid = labels >= 0
    ks = sorted(int(c) for c in np.unique(labels[valid]))
    cent = np.vstack([H[labels == c].mean(0) for c in ks])
    D = np.sqrt(((H[:, None, :] - cent[None, :, :]) ** 2).sum(2))
    o = np.argsort(D, axis=1); r = np.arange(len(D))
    d1, d2 = D[r, o[:, 0]], D[r, o[:, 1]]
    return np.where(valid, (d2 - d1) / (d2 + d1 + 1e-12), np.nan)


def select_k(frac, valid, k_range=(3, 4, 5, 6), seed=42, min_ari=0.80, sil_tol=0.02, n_sub=30000):
    """Pick the smallest k that is split-half stable and within sil_tol of the best silhouette."""
    rng = np.random.default_rng(seed); H = hellinger(frac); vidx = np.where(valid)[0]
    sub = rng.choice(vidx, min(n_sub, len(vidx)), replace=False)
    perm = rng.permutation(vidx); third = len(perm) // 3
    ia, ib, ic = perm[:third], perm[third:2 * third], perm[2 * third:]
    rows = []
    for k in k_range:
        km = KMeans(k, n_init=5, random_state=seed).fit(H[sub])
        sil = silhouette_score(H[sub], km.labels_, sample_size=min(10000, len(sub)), random_state=seed)
        a = KMeans(k, n_init=5, random_state=1).fit(H[ia]); b = KMeans(k, n_init=5, random_state=2).fit(H[ib])
        rows.append((k, sil, adjusted_rand_score(a.predict(H[ic]), b.predict(H[ic]))))
    ok = [r for r in rows if r[2] >= min_ari] or rows
    best = max(r[1] for r in ok)
    return int(min(r[0] for r in ok if r[1] >= best - sil_tol)), rows
