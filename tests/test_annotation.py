import numpy as np

from spatialml import annotation as an


def _expr(seed=0):
    rng = np.random.default_rng(seed)
    genes = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"]
    clusters = np.repeat(np.arange(4), 300)
    X = rng.poisson(0.2, (1200, 8)).astype(float)
    X[clusters == 0, 0:2] += 3; X[clusters == 1, 2:4] += 3; X[clusters == 2, 4:6] += 3
    X[clusters == 2, 6:8] += 3              # cluster 2 expresses lineage C AND D equally -> ambiguous
    return np.log1p(X), clusters, genes


def test_annotation_and_ambiguity():
    X, cl, genes = _expr()
    means, sizes = an.cluster_mean_expression(X, cl, genes)
    markers = {"LinA": ["A1", "A2"], "LinB": ["B1", "B2"], "LinC": ["C1", "C2"], "LinD": ["D1", "D2"]}
    table, _ = an.annotate_clusters(means, sizes, markers)
    assert table.loc["0", "label"] == "LinA" and table.loc["1", "label"] == "LinB"
    assert not table.loc["2", "confident"] and table.loc["2", "label"] == "Ambiguous"


def test_refinement_merges_only_compatible_pairs():
    import pandas as pd
    t = pd.DataFrame({"confident": [False, False, True], "label": ["Ambiguous", "Ambiguous", "T_cell"],
                      "top_lineage": ["Macrophage_myeloid", "T_cell", "T_cell"],
                      "second_lineage": ["Dendritic_cell", "Macrophage_myeloid", "B_cell"], "top_score": [0.9, 0.9, 0.9]}, index=["a", "b", "c"])
    r = an.refine_labels(t)
    assert r.loc["a", "fine_label"] == "Myeloid" and r.loc["b", "fine_label"] == "Unassigned" and r.loc["c", "fine_label"] == "T_cell"
    assert list(an.to_coarse(["Myeloid", "Unassigned", "Stromal_adipose"])) == ["Myeloid", "Unassigned", "Stromal"]
