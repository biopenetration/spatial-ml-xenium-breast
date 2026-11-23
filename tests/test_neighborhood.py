import numpy as np
import pytest
from scipy.spatial import cKDTree

from spatialml import neighborhood as nbh


@pytest.fixture(scope="module")
def tissue():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 600, (1500, 2)); codes = rng.integers(0, 4, 1500)
    return xy, codes, np.eye(4, dtype=np.float32)[codes]


def test_ring_stats_matches_brute_force(tissue):
    xy, codes, onehot = tissue
    rings = [(0, 15), (0, 30), (50, 100)]
    st = nbh.ring_stats(xy, onehot, rings, chunk=400)          # small chunk exercises chunk borders
    for i in np.random.default_rng(1).choice(len(xy), 120, replace=False):
        d = np.hypot(*(xy - xy[i]).T)
        for r_in, r_out in rings:
            m = (d >= r_in) & (d <= r_out); m[i] = False         # self is always excluded
            assert st[(r_in, r_out)]["n"][i] == m.sum()
            assert np.allclose(st[(r_in, r_out)]["counts"][i], onehot[m].sum(0))


def test_entropy_bounds_and_nan():
    h = nbh.shannon_entropy(np.array([[5, 0, 0], [1, 1, 1], [0, 0, 0]], float))
    assert np.isclose(h[0], 0) and np.isclose(h[1], np.log(3)) and np.isnan(h[2])


def test_nearest_type_distance(tissue):
    xy, codes, _ = tissue
    d = nbh.nearest_type_distance(xy, codes, 2)
    for i in range(0, 1500, 150):
        m = codes == 2; m[i] = False
        assert np.isclose(d[i], np.hypot(*(xy[m] - xy[i]).T).min())


def test_spatial_folds_buffer_and_balance(tissue):
    xy, _, _ = tissue
    block, fold = nbh.make_spatial_folds(xy, 150, 4, seed=0)
    sizes = np.bincount(fold); assert sizes.max() / sizes.min() < 2.0
    for f in range(4):
        tr, te = nbh.buffered_train_mask(xy, fold, f, 60)
        assert not (tr & te).any()
        dmin, _ = cKDTree(xy[te]).query(xy[tr], k=1)
        assert dmin.min() >= 60 - 1e-9
    for b in np.unique(block):                                   # a block never spans two folds
        assert len(np.unique(fold[block == b])) == 1


def test_nearest_type_distance_with_a_single_cell_of_the_type():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 25.0], [100.0, 100.0]])
    codes = np.array([0, 0, 0, 1])                       # type 1 has exactly one cell
    d = nbh.nearest_type_distance(xy, codes, 1)
    assert np.isnan(d[3]) and np.allclose(d[:3], np.hypot(xy[:3, 0] - 100, xy[:3, 1] - 100))
    assert np.isnan(nbh.nearest_type_distance(xy, codes, 2)).all()                 # absent type -> NaN
