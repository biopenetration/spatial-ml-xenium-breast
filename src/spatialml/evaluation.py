"""spatialml.evaluation - split builders and a generic cross-validation runner (Module 6)."""
import numpy as np
from . import neighborhood as nbh
from . import modeling as ml


def spatial_splits(xy, valid, fold, buffer_um, max_train=None, seed=0):
    """Block folds; training cells closer than buffer_um to any test cell are dropped. valid = cells with a target."""
    out = {}
    for f in sorted(np.unique(fold)):
        tr_m, te_m = nbh.buffered_train_mask(xy, fold, f, buffer_um)
        tr = np.where(tr_m & valid)[0]; te = np.where(te_m & valid)[0]
        if max_train and len(tr) > max_train:
            tr = np.sort(np.random.default_rng(seed + int(f)).choice(tr, max_train, replace=False))
        out[int(f)] = (tr, te)
    return out


def random_splits(valid, n_folds=5, max_train=None, seed=0):
    """Ordinary random K-fold over cells (NO spatial separation): used only to demonstrate leakage."""
    rng = np.random.default_rng(seed)
    idx = np.where(valid)[0]; rng.shuffle(idx)
    parts = np.array_split(idx, n_folds); out = {}
    for f, te in enumerate(parts):
        tr = np.concatenate([p for g, p in enumerate(parts) if g != f])
        if max_train and len(tr) > max_train:
            tr = np.sort(rng.choice(tr, max_train, replace=False))
        out[f] = (tr, np.sort(te))
    return out


def run_cv(X, y, splits, cols, kind, params, seed, K):
    """Out-of-fold class probabilities (n, K); NaN rows were never in a test fold."""
    proba = np.full((len(X), K), np.nan, np.float32)
    for f, (tr, te) in splits.items():
        m = ml.build_model(kind, cols, seed, params).fit(X.iloc[tr][cols], y[tr])
        pf = np.zeros((len(te), K), np.float32)
        pf[:, np.asarray(m.classes_, dtype=int)] = m.predict_proba(X.iloc[te][cols])
        proba[te] = pf
    return proba
