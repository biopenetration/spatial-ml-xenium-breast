import numpy as np
import pytest

from spatialml import features as ft
from spatialml.annotation import COARSE_TYPES
from spatialml.demo import make_synthetic_tissue


@pytest.fixture(scope="module")
def tissue():
    return make_synthetic_tissue(2500, seed=1)


def test_feature_names_and_no_inf(tissue):
    f = ft.build_spatial_features(tissue[["x", "y"]].to_numpy(), tissue["cell_type"].to_numpy())
    assert "frac_Epithelial_r30" in f and "entropy_r15" in f and "knn_dist_cap30_k10" in f
    assert all(f"dist_nearest_{t}_cap30" in f for t in COARSE_TYPES)
    assert np.isfinite(f.to_numpy()[~np.isnan(f.to_numpy())]).all()


def test_capped_distances_never_exceed_cap(tissue):
    f = ft.build_spatial_features(tissue[["x", "y"]].to_numpy(), tissue["cell_type"].to_numpy(), cap_um=30)
    dist_cols = [c for c in f if c.startswith(("dist_nearest_", "knn_dist_"))]
    assert (f[dist_cols].max() <= 30 + 1e-9).all()


def test_reach_guarantee_by_deleting_distant_cells(tissue):
    """Deleting every cell farther than the cap from a focal cell must not change any of its features."""
    worst = ft.reach_test(tissue[["x", "y"]].to_numpy(), tissue["cell_type"].to_numpy(), n_cells=25, seed=0)
    assert worst < 1e-6


def test_radius_above_cap_is_rejected(tissue):
    with pytest.raises(ValueError):
        ft.build_spatial_features(tissue[["x", "y"]].to_numpy(), tissue["cell_type"].to_numpy(), radii=(15, 60), cap_um=30)


def test_density_and_counts_consistent(tissue):
    f = ft.build_spatial_features(tissue[["x", "y"]].to_numpy(), tissue["cell_type"].to_numpy())
    assert np.allclose(f["density_r30"], f["n_cells_r30"] / (np.pi * 0.03 ** 2))
    frac_cols = [c for c in f if c.startswith("frac_") and c.endswith("_r30")]
    s = f[frac_cols].sum(axis=1).to_numpy(); ok = f["n_cells_r30"].to_numpy() > 0
    assert np.allclose(s[ok], 1.0)
