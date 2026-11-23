import numpy as np
import pytest

from spatialml import targets as tg
from spatialml.annotation import COARSE_TYPES
from spatialml.config import load_config
from spatialml.demo import make_synthetic_tissue


@pytest.fixture(scope="module")
def data():
    t = make_synthetic_tissue(3500, seed=2)
    xy = t[["x", "y"]].to_numpy(); ct = t["cell_type"].to_numpy()
    frac, n_ann, valid = tg.annulus_composition(xy, ct, (50, 100), min_cells=10)
    return xy, ct, frac, n_ann, valid


def test_annulus_matches_brute_force(data):
    xy, ct, frac, n_ann, valid = data
    for i in np.where(valid)[0][::200]:
        d = np.hypot(*(xy - xy[i]).T); m = (d >= 50) & (d <= 100); m[i] = False
        assert n_ann[i] == m.sum()
        counts = np.array([(ct[m] == t).sum() for t in COARSE_TYPES], float)
        assert np.allclose(frac[i], counts / counts.sum())


def test_classes_names_margin(data):
    xy, ct, frac, n_ann, valid = data
    labels = tg.fit_target_classes(frac, valid, k=3, seed=0)
    assert (labels[~valid] == -1).all() and set(np.unique(labels[valid])) == {0, 1, 2}
    again = tg.fit_target_classes(frac, valid, k=3, seed=0)
    assert np.array_equal(labels, again)                                  # deterministic
    names = tg.name_classes(frac, labels)
    assert len(set(names.values())) == 3 and all(n.startswith("ME") for n in names.values())
    m = tg.class_margin(frac, labels)
    assert np.nanmin(m) >= 0 and np.nanmax(m) <= 1 and np.isnan(m[~valid]).all()


def test_select_k_returns_value_in_range(data):
    _, _, frac, _, valid = data
    k, rows = tg.select_k(frac, valid, k_range=(2, 3, 4), n_sub=2000)
    assert k in (2, 3, 4) and len(rows) == 3


def test_leakage_rules_in_config():
    cfg = load_config()                                                   # defaults obey the rules
    assert cfg["features"]["cap_um"] < cfg["target"]["annulus_um"][0]
    assert cfg["cv"]["buffer_um"] >= cfg["target"]["annulus_um"][1]
    with pytest.raises(ValueError):
        load_config(overrides={"features": {"cap_um": 60, "radii_um": [15, 30]}})     # cap reaches into the target ring
    with pytest.raises(ValueError):
        load_config(overrides={"cv": {"buffer_um": 50}})                              # buffer smaller than target ring
