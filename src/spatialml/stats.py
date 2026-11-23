"""spatialml.stats - rank-based effect sizes, block bootstrap and spatial null models (Module 4)."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def rank_matrix(df):
    """Per-column percentile ranks (rank / n_valid, ties averaged). Returns P (float64, NaN->0) and mask M (1 = observed).
    float64 matters: eta-squared is a small difference of large sums, so float32 ranks lose precision."""
    P = df.rank(method="average", pct=True)
    M = P.notna().to_numpy(np.float32)
    return P.fillna(0.0).to_numpy(np.float64), M


def group_sums(P, M, lab, K, group=None, n_groups=1):
    """Sums of ranks per (group, class, feature). lab = class index (-1 = excluded); group = e.g. spatial block."""
    n = len(lab)
    idx = np.where(lab >= 0)[0]
    g = np.zeros(n, np.int64) if group is None else np.asarray(group, np.int64)
    rows = g[idx] * K + lab[idx]
    Y = sp.csr_matrix((np.ones(len(idx)), (rows, idx)), shape=(n_groups * K, n))
    S = np.asarray(Y @ P, dtype=np.float64).reshape(n_groups, K, -1)
    N = np.asarray(Y @ M, dtype=np.float64).reshape(n_groups, K, -1)
    Wg = sp.csr_matrix((np.ones(len(idx)), (g[idx], idx)), shape=(n_groups, n))
    Q = np.asarray(Wg @ (P.astype(np.float64) ** 2))
    return S, N, Q


def eta2_from_sums(S, N, Q):
    """Rank eta-squared (= Kruskal-Wallis H / (N-1)). Leading axes are broadcast (e.g. bootstrap replicates)."""
    St, Nt = S.sum(-2), N.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ssb = (S ** 2 / np.where(N > 0, N, np.inf)).sum(-2) - St ** 2 / Nt
        sst = Q - St ** 2 / Nt
        return ssb / sst


def auc_from_sums(S, N):
    """One-vs-rest AUC per class: P(feature value of a class-c cell > a cell outside c)."""
    St, Nt = S.sum(-2, keepdims=True), N.sum(-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_c = S / np.where(N > 0, N, np.nan)
        rest = (St - S) / np.where(Nt - N > 0, Nt - N, np.nan)
    return 0.5 + mean_c - rest


def block_bootstrap(S_b, N_b, Q_b, n_boot, rng):
    """Resample spatial blocks with replacement. Inputs have a leading block axis."""
    nb = S_b.shape[0]
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot).astype(np.float64)
    S = np.einsum("rb,bkp->rkp", W, S_b)
    N = np.einsum("rb,bkp->rkp", W, N_b)
    Q = W @ Q_b
    return eta2_from_sums(S, N, Q), auc_from_sums(S, N)


def shift_null_labels(xy, lab, rng, min_shift_um=1000.0, max_assign_um=30.0):
    """Toroidal-shift null: slide the label map over the tissue by a random vector (wrapping at the bounding box).
    Preserves the spatial autocorrelation of labels and features, breaks their cross-association."""
    mn = xy.min(0); ext = xy.max(0) - mn
    for _ in range(1000):
        s = rng.uniform(0, 1, 2) * ext
        if np.hypot(*s) >= min_shift_um: break
    shifted = (xy - mn + s) % ext + mn
    v = lab >= 0
    d, j = cKDTree(shifted[v]).query(xy, k=1, workers=-1)
    return np.where(d <= max_assign_um, lab[v][j], -1)


def benjamini_hochberg(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0)
    return out


def within_group_permutation(codes, group, rng):
    """Shuffle labels only inside each group (e.g. spatial tile): keeps regional composition."""
    order = np.lexsort((rng.random(len(codes)), group))
    base = np.argsort(group, kind="stable")
    out = np.empty_like(codes)
    out[base] = codes[order]
    return out


def pair_counts(ci, cj, T):
    """Counts of unordered type pairs; returns a (T, T) matrix filled on the upper triangle."""
    lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
    return np.bincount(lo * T + hi, minlength=T * T).reshape(T, T)
