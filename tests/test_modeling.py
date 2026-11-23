import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score

from spatialml import modeling as ml


def test_block_count_metrics_match_sklearn():
    rs = np.random.default_rng(0); K = 4
    yt = rs.integers(0, K, 5000); yp = np.where(rs.random(5000) < .6, yt, rs.integers(0, K, 5000)); blk = rs.integers(0, 50, 5000)
    m = ml.metrics_from_counts(ml.block_confusions(yt, yp, blk, 50, K).sum(0))
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(yt, yp))
    assert np.isclose(m["macro_f1"], f1_score(yt, yp, average="macro"))
    assert np.isclose(m["accuracy"], accuracy_score(yt, yp))
    y = rs.random(5000); p = y + rs.normal(0, .2, 5000)
    r2, rmse = ml.r2_rmse_from_sums(ml.block_regression_sums(y, p, blk, 50).sum(0))
    assert np.isclose(r2, r2_score(y, p)) and np.isclose(rmse, mean_squared_error(y, p) ** 0.5)


def test_type_lookup_probabilities_and_prediction():
    cols = ["type_a", "type_b"]
    X = pd.DataFrame({"type_a": [1, 1, 1, 0, 0, 0], "type_b": [0, 0, 0, 1, 1, 1]}); y = np.array([0, 0, 1, 1, 1, 0])
    m = ml.TypeLookup(type_cols=cols).fit(X, y)
    p = m.predict_proba(X)
    assert np.allclose(p.sum(1), 1) and list(m.predict(X)) == [0, 0, 0, 1, 1, 1]


def test_models_fit_predict_with_missing_values():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 6)), columns=["dist_nearest_a", "knn_dist_x", "f1", "f2", "f3", "f4"])
    X[["dist_nearest_a", "knn_dist_x"]] = np.abs(X[["dist_nearest_a", "knn_dist_x"]])      # distances are non-negative
    y = (X["f1"] + X["f2"] > 0).astype(int).to_numpy(); X.iloc[::17, 2] = np.nan
    for kind in ["logreg", "rf", "hgb"]:
        m = ml.build_model(kind, list(X.columns), 0, {"n_estimators": 30, "max_iter": 30}).fit(X, y)
        assert balanced_accuracy_score(y, m.predict(X)) > 0.8
