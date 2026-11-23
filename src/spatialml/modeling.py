"""spatialml.modeling - leakage-safe model builders, block-bootstrap metrics, group permutation importance (Module 5)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LOG_PREFIXES = ("dist_nearest_", "knn_dist_", "mean_nbr_dist_", "n_cells_", "density_")


class TypeLookup(BaseEstimator, ClassifierMixin):
    """Class probabilities from the cell's own type only (one-hot columns). The baseline spatial features must beat."""
    def __init__(self, type_cols=None, alpha=1.0):
        self.type_cols = type_cols; self.alpha = alpha

    def fit(self, X, y):
        cols = list(self.type_cols); y = np.asarray(y)
        self.classes_ = np.unique(y); K = len(self.classes_)
        t = pd.DataFrame(X)[cols].to_numpy().argmax(1)
        counts = np.zeros((len(cols), K))
        np.add.at(counts, (t, np.searchsorted(self.classes_, y)), 1)
        self.table_ = (counts + self.alpha) / (counts.sum(1, keepdims=True) + self.alpha * K)
        return self

    def predict_proba(self, X):
        return self.table_[pd.DataFrame(X)[list(self.type_cols)].to_numpy().argmax(1)]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class TypeMean(BaseEstimator, RegressorMixin):
    """Mean of a continuous target per own cell type (regression baseline)."""
    def __init__(self, type_cols=None):
        self.type_cols = type_cols

    def fit(self, X, y):
        cols = list(self.type_cols); y = np.asarray(y, float)
        t = pd.DataFrame(X)[cols].to_numpy().argmax(1)
        self.means_ = np.array([y[t == k].mean() if (t == k).any() else y.mean() for k in range(len(cols))])
        return self

    def predict(self, X):
        return self.means_[pd.DataFrame(X)[list(self.type_cols)].to_numpy().argmax(1)]


def _linear_prep(cols):
    logc = [c for c in cols if c.startswith(LOG_PREFIXES)]
    rest = [c for c in cols if c not in logc]
    blocks = []
    if logc:
        blocks.append(("log", Pipeline([("l", FunctionTransformer(np.log1p)),
                                        ("i", SimpleImputer(strategy="median", add_indicator=True)),
                                        ("s", StandardScaler())]), logc))
    if rest:
        blocks.append(("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                                        ("s", StandardScaler())]), rest))
    return ColumnTransformer(blocks, remainder="drop")


def build_model(kind, cols, seed=0, params=None):
    """All preprocessing lives inside the estimator, so it is fit on training cells only."""
    p = dict(params or {})
    if kind == "dummy":
        return DummyClassifier(strategy="prior")
    if kind == "lookup":
        return TypeLookup(type_cols=p["type_cols"])
    if kind == "typemean":
        return TypeMean(type_cols=p["type_cols"])
    if kind == "logreg":
        return Pipeline([("prep", _linear_prep(cols)),
                         ("m", LogisticRegression(C=p.get("C", 1.0), max_iter=500, class_weight="balanced"))])
    if kind == "ridge":
        return Pipeline([("prep", _linear_prep(cols)), ("m", Ridge(alpha=p.get("alpha", 10.0)))])
    if kind == "rf":
        return Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                         ("m", RandomForestClassifier(n_estimators=p.get("n_estimators", 200), min_samples_leaf=p.get("min_samples_leaf", 20),
                                                      max_features="sqrt", max_samples=p.get("max_samples", 0.4),
                                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed))])
    hp = dict(learning_rate=p.get("learning_rate", 0.1), max_iter=p.get("max_iter", 200), max_leaf_nodes=p.get("max_leaf_nodes", 15),
              l2_regularization=p.get("l2_regularization", 1.0), min_samples_leaf=p.get("min_samples_leaf", 50),
              early_stopping=False, random_state=seed)
    if kind == "hgb":
        return HistGradientBoostingClassifier(class_weight="balanced", **hp)
    if kind == "hgbr":
        return HistGradientBoostingRegressor(**hp)
    raise ValueError(kind)


def tune_hgb(X, y, groups, grid, seed=0, n_splits=3, max_cells=30000):
    """Pick HGB hyper-parameters with an inner GroupKFold over spatial blocks (training cells only)."""
    rng = np.random.default_rng(seed)
    if len(X) > max_cells:
        s = rng.choice(len(X), max_cells, replace=False); X, y, groups = X.iloc[s], np.asarray(y)[s], np.asarray(groups)[s]
    best = (-1.0, grid[0])
    for params in grid:
        sc = []
        for tr, te in GroupKFold(n_splits).split(X, y, groups):
            m = build_model("hgb", list(X.columns), seed, params).fit(X.iloc[tr], y[tr])
            sc.append(balanced_accuracy_score(y[te], m.predict(X.iloc[te])))
        if np.mean(sc) > best[0]: best = (float(np.mean(sc)), params)
    return best[1], best[0]


# ---------- metrics computed from block-level confusion matrices (fast block bootstrap) ----------
def block_confusions(y, pred, blk, nb, K):
    idx = (np.asarray(blk) * K + np.asarray(y)) * K + np.asarray(pred)
    return np.bincount(idx, minlength=nb * K * K).reshape(nb, K, K).astype(np.float64)


def metrics_from_counts(C):
    C = np.asarray(C, float)
    tp = np.einsum("...ii->...i", C); support = C.sum(-1); predicted = C.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = tp / support; f1 = 2 * tp / (support + predicted)
        return {"balanced_accuracy": np.nanmean(recall, -1), "macro_f1": np.nanmean(f1, -1),
                "accuracy": tp.sum(-1) / C.sum((-1, -2)), "recall": recall}


def bootstrap_weights(nb, n_boot, rng):
    return rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot).astype(np.float64)


def block_regression_sums(y, pred, blk, nb):
    y = np.asarray(y, float); pred = np.asarray(pred, float); blk = np.asarray(blk)
    out = np.zeros((nb, 4))
    out[:, 0] = np.bincount(blk, minlength=nb); out[:, 1] = np.bincount(blk, weights=y, minlength=nb)
    out[:, 2] = np.bincount(blk, weights=y * y, minlength=nb); out[:, 3] = np.bincount(blk, weights=(y - pred) ** 2, minlength=nb)
    return out


def r2_rmse_from_sums(S):
    n, sy, syy, sse = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    return 1 - sse / (syy - sy ** 2 / n), np.sqrt(sse / n)


def group_permutation_importance(model, X, y, groups, rng, n_repeats=3, metric=balanced_accuracy_score):
    """Drop in `metric` when all columns of a group are permuted jointly (same row permutation)."""
    base = metric(y, model.predict(X)); out = {}
    for g, cols in groups.items():
        arr = X[cols].to_numpy(); drops = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[cols] = arr[rng.permutation(len(X))]
            drops.append(base - metric(y, model.predict(Xp)))
        out[g] = (float(np.mean(drops)), float(np.std(drops)))
    return base, out
