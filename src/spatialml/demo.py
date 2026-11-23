"""spatialml.demo - a small synthetic tissue so the whole pipeline can run (and be tested) without the real data."""
import numpy as np
import pandas as pd

from .annotation import COARSE_TYPES

_MIX = {  # region -> probability of each coarse type
    0: [.80, .03, .00, .05, .00, .02, .05, .05],
    1: [.60, .15, .00, .05, .00, .03, .07, .10],
    2: [.10, .30, .05, .20, .01, .04, .10, .20],
    3: [.05, .10, .03, .12, .01, .09, .20, .40],
    4: [.15, .05, .00, .10, .00, .10, .25, .35],
    5: [.70, .05, .00, .05, .00, .02, .08, .10],
}


def make_synthetic_tissue(n_cells=6000, density_per_mm2=4000, seed=0):
    """Random cell positions in a square tissue; six Voronoi-like regions with different cell-type mixtures."""
    rng = np.random.default_rng(seed)
    side = float(np.sqrt(n_cells / density_per_mm2) * 1000.0)
    xy = rng.uniform(0, side, (n_cells, 2))
    centers = rng.uniform(0, side, (len(_MIX), 2))
    region = np.argmin(((xy[:, None, :] - centers[None]) ** 2).sum(2), axis=1)
    types = np.empty(n_cells, dtype=object)
    for r, p in _MIX.items():
        m = region == r
        types[m] = rng.choice(COARSE_TYPES, size=int(m.sum()), p=np.array(p) / sum(p))
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cell_type": types, "region": region})
