import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score

from spatialml import stats as st


def _data(n=3000, K=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    X = pd.DataFrame({"a": rng.normal(y, 1.0), "b": rng.normal(0, 1, n), "c": np.round(rng.normal(y * 0.3, 1, n), 1)})
    X.loc[rng.choice(n, 100, replace=False), "b"] = np.nan                     # NaNs and ties are handled
    return X, y, K


def test_eta2_equals_kruskal_and_auc_equals_sklearn():
    X, y, K = _data()
    P, M = st.rank_matrix(X)
    S, N, Q = st.group_sums(P, M, y, K)
    eta = st.eta2_from_sums(S[0], N[0], Q[0]); auc = st.auc_from_sums(S[0], N[0])
    for j, c in enumerate(X.columns):
        x = X[c].to_numpy(float); ok = ~np.isnan(x)
        H = kruskal(*[x[ok & (y == k)] for k in range(K)]).statistic
        assert np.isclose(eta[j], H / (ok.sum() - 1), rtol=1e-6)
        for k in range(K):
            assert np.isclose(auc[k, j], roc_auc_score((y[ok] == k).astype(int), x[ok]), atol=1e-6)


def test_block_bootstrap_ci_contains_estimate():
    X, y, K = _data(4000)
    blk = np.random.default_rng(1).integers(0, 40, len(y))
    P, M = st.rank_matrix(X)
    S, N, Q = st.group_sums(P, M, y, K, group=blk, n_groups=40)
    eta = st.eta2_from_sums(S.sum(0), N.sum(0), Q.sum(0))
    boot, _ = st.block_bootstrap(S, N, Q, 300, np.random.default_rng(2))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    assert np.all(lo <= eta + 1e-9) and np.all(eta <= hi + 1e-9)


def test_benjamini_hochberg_monotone():
    p = np.array([0.001, 0.01, 0.03, 0.04, 0.5]); q = st.benjamini_hochberg(p)
    assert np.all(q >= p) and np.all(np.diff(q[np.argsort(p)]) >= -1e-12) and q.max() <= 1


def test_within_group_permutation_preserves_group_counts():
    rng = np.random.default_rng(0); g = rng.integers(0, 15, 3000); c = rng.integers(0, 4, 3000)
    c2 = st.within_group_permutation(c, g, rng)
    for k in range(15):
        assert np.array_equal(np.bincount(c[g == k], minlength=4), np.bincount(c2[g == k], minlength=4))
    assert (c != c2).mean() > 0.3


def test_shift_null_keeps_label_distribution():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1500, (8000, 2)); lab = (xy[:, 0] > 750).astype(int)
    new = st.shift_null_labels(xy, lab, rng, min_shift_um=500, max_assign_um=30)
    kept = new >= 0
    assert kept.mean() > 0.9 and abs(new[kept].mean() - lab.mean()) < 0.1
