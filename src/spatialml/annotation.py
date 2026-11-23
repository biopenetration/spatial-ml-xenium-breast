"""spatialml.annotation - marker-based cluster annotation with explicit uncertainty, and documented label refinement."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

COARSE_TYPES = ["Epithelial", "T_cell", "B_cell", "Myeloid", "Mast_cell", "Endothelial", "Mural", "Stromal"]

LINEAGE_TO_COARSE = {
    "Epithelial_luminal": "Epithelial", "Epithelial_basal_luminal": "Epithelial", "Myoepithelial_basal": "Epithelial",
    "T_cell": "T_cell", "B_cell": "B_cell", "Mast_cell": "Mast_cell", "Endothelial": "Endothelial",
    "Macrophage_myeloid": "Myeloid", "Dendritic_cell": "Myeloid", "Myeloid": "Myeloid",
    "Pericyte_smooth_muscle": "Mural", "Mural_myoepithelial": "Mural",
    "Fibroblast_stromal": "Stromal", "Stromal_adipose": "Stromal", "Adipocyte": "Stromal",
}

# An ambiguous cluster is resolved ONLY when its top two lineages are a biologically compatible pair.
COMPATIBLE_PAIRS = [
    ({"Macrophage_myeloid", "Dendritic_cell"}, "Myeloid"),
    ({"Myoepithelial_basal", "Pericyte_smooth_muscle"}, "Mural_myoepithelial"),
    ({"Myoepithelial_basal", "Epithelial_luminal"}, "Epithelial_basal_luminal"),
    ({"Adipocyte", "Fibroblast_stromal"}, "Stromal_adipose"),
]


def cluster_mean_expression(expr, clusters, gene_names):
    """Mean expression per cluster. expr: (cells x genes) dense or sparse; returns (DataFrame, sizes Series)."""
    cl = pd.Categorical(np.asarray(clusters).astype(str))
    codes, k, n = cl.codes, len(cl.categories), expr.shape[0]
    ind = sp.csr_matrix((np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(k, n))
    sums = ind @ expr
    sums = sums.toarray() if sp.issparse(sums) else np.asarray(sums)
    sizes = np.bincount(codes, minlength=k)
    return pd.DataFrame(sums / sizes[:, None], index=cl.categories, columns=list(gene_names)), pd.Series(sizes, index=cl.categories, name="n_cells")


def annotate_clusters(means, sizes, markers, min_top=0.35, amb_ratio=0.80):
    """Score each cluster for each lineage (min-max normalised mean marker expression); flag ambiguous clusters."""
    present = {k: [g for g in v if g in means.columns] for k, v in markers.items()}
    present = {k: v for k, v in present.items() if len(v) >= 2}
    span = means.max(axis=0) - means.min(axis=0)
    norm = (means - means.min(axis=0)) / span.where(span > 0.05, np.inf)
    scores = pd.DataFrame({lin: norm[g].mean(axis=1) for lin, g in present.items()})
    arr = scores.to_numpy(); order = np.argsort(-arr, axis=1); r = np.arange(len(scores))
    table = pd.DataFrame({"n_cells": sizes, "top_lineage": scores.columns[order[:, 0]], "top_score": arr[r, order[:, 0]],
                          "second_lineage": scores.columns[order[:, 1]], "second_score": arr[r, order[:, 1]]}, index=scores.index)
    table["score_ratio"] = table["second_score"] / table["top_score"].clip(lower=1e-9)
    table["confident"] = (table["top_score"] >= min_top) & (table["score_ratio"] < amb_ratio)
    table["label"] = np.where(table["confident"], table["top_lineage"], "Ambiguous")
    return table, scores


def refine_labels(table, min_top=0.40):
    """Resolve ambiguous clusters that are compatible lineage pairs; everything else becomes 'Unassigned'."""
    fine, rule = [], []
    for _, row in table.iterrows():
        if bool(row["confident"]):
            fine.append(row["label"]); rule.append("confident")
            continue
        pair = {row["top_lineage"], row["second_lineage"]}
        for members, name in COMPATIBLE_PAIRS:
            if pair == members and row["top_score"] >= min_top:
                fine.append(name); rule.append("resolved: compatible pair"); break
        else:
            fine.append("Unassigned"); rule.append("kept ambiguous")
    out = table.copy(); out["fine_label"] = fine; out["rule"] = rule
    return out


def to_coarse(fine_labels):
    return pd.Series(fine_labels).map(LINEAGE_TO_COARSE).fillna("Unassigned").to_numpy()
