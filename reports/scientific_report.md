# Predicting tumour microenvironment class from spatial neighbourhoods in Xenium breast cancer data

## Abstract
We asked whether the cellular neighbourhood of a cell (cells within 30 um) predicts the composition class of the surrounding tissue (a 50-100 um ring), beyond what the cell's own transcriptome or cell type reveals.
Using a public Xenium FFPE human breast cancer section (164,000 cells, 313 genes) and spatial block cross-validation with a 100 um train/test buffer, leakage-controlled spatial features reached a balanced accuracy of 0.746 (95% CI 0.733-0.756), versus 0.612 (95% CI 0.602-0.621) for an own-cell-type baseline and 0.731 (95% CI 0.717-0.745) for expression alone.
Combining spatial and expression features changed balanced accuracy relative to expression alone by +0.042 (95% CI +0.035 to +0.049). An early analysis with unbounded nearest-type distances reached 0.833 (95% CI 0.818-0.846); this was identified as leakage (the features reached into the target ring) and corrected (difference -0.087 (95% CI -0.098 to -0.077)). Results were stable across buffer size, block size and partition seed; accuracy decreased monotonically as the target ring moved away from the cell but stayed above the own-type baseline. All results come from one section and are exploratory.

## 1. Data and quality control
Public 10x Genomics Xenium FFPE Human Breast Cancer (Rep 1). Cell identifiers and coordinates were validated, cells with fewer than 10 counts or 3 detected genes were removed, and raw counts were kept in a separate layer. 164,000 cells and 313 genes were analysed.

## 2. Cell-type labels
Cells were clustered (Leiden, several resolutions; 21 clusters at the primary resolution) and clusters were scored against marker lists for 11 lineages restricted to genes on the panel. A cluster was labelled only if its best lineage score was high and clearly above the runner-up; otherwise it was flagged ambiguous (6 clusters, 16.9% of cells). Ambiguous clusters were then resolved only when their top two lineages were a biologically compatible pair (for example macrophage and dendritic cell); the rest stayed unassigned (0.8% of cells) and were excluded from targets. Annotation agreement across resolutions and seeds was at least 0.84, and a classifier that never saw the marker genes recovered the labels with balanced accuracy 0.91. Eight coarse types were used for composition features.

## 3. Features, target and evaluation
* **Features (40).** Composition, density, diversity, edge-asymmetry and capped distance features computed from cells within 30 um, excluding the focal cell. Nearest-type distances were capped at 30 um.
* **Target.** Composition of assigned cells in the 50-100 um ring, Hellinger-transformed and clustered (k-means; k = 3 chosen by split-half stability and parsimony) into 3 classes: `ME1_Epithelial-rich`, `ME2_Epithelial/T_cell`, `ME3_T_cell/Stromal`. The classes form a gradient rather than sharply separated clusters (silhouette 0.34), but they are spatially coherent: 0.94 of a cell's 10 nearest neighbours share its class versus 0.35 expected by chance. The class margin (distance to the second-closest centroid) flags boundary cells.
* **Evaluation.** Tissue blocks of 500 um were assigned to folds; training cells within 100 um of any test cell were removed; preprocessing was fitted inside each fold. Confidence intervals resample blocks; model comparisons are paired.

## 4. Results
| Features | Balanced accuracy (95% CI) |
|---|---|
| Own cell type only (cross-validated baseline) | 0.612 (95% CI 0.602-0.621) |
| Coordinates (x, y) only | 0.572 (95% CI 0.536-0.611) |
| Long-range distances only (leakage probe) | 0.825 (95% CI 0.810-0.839) |
| Spatial, original (unbounded distances; **leaky**) | 0.833 (95% CI 0.818-0.846) |
| Spatial, strict (reach <= 30 um) | 0.746 (95% CI 0.733-0.756) |
| Strict spatial, 15 um features only | 0.669 (95% CI 0.658-0.678) |
| Strict spatial, 30 um + capped distances | 0.745 (95% CI 0.732-0.756) |
| Strict neighbour composition only | 0.746 (95% CI 0.733-0.756) |
| Expression only (own transcriptome) | 0.731 (95% CI 0.717-0.745) |
| Expression excluding marker genes | 0.711 (95% CI 0.697-0.725) |
| Strict spatial + expression | 0.773 (95% CI 0.759-0.785) |

* **Beyond own cell type.** Strict spatial features versus the own-type baseline: +0.133 (95% CI +0.124 to +0.143).
* **Spatial vs expression.** Spatial minus expression-only: +0.014 (95% CI +0.004 to +0.024). Combined minus expression-only: +0.042 (95% CI +0.035 to +0.049) - spatial features add information beyond the transcriptome in this section. Combined minus spatial-only: +0.027 (95% CI +0.022 to +0.033), so expression also adds information beyond the neighbourhood.
* **Not just marker genes.** Expression without the genes used to define cell types: 0.711 (95% CI 0.697-0.725).
* **Statistical association (Notebook 04).** 31 of 43 features were associated with the class above a spatial (toroidal-shift) null model, whereas a naive test flagged 43 of 43. Within a given cell type the association shrinks markedly (median rank eta-squared Epithelial 0.013, Myeloid 0.023, Stromal 0.050, T_cell 0.029, against 0.124 overall), consistent with much of the signal being explained by cell identity. 65 of 72 cell-type pair enrichment tests survived a within-tile permutation null, i.e. they reflect local organisation beyond regional composition. Morphology features (a planned negative control) were not null (mean eta-squared 0.098; 2/3 supported), plausibly because cell size differs by cell type and local density, so they were kept out of the spatial feature set.

## 5. Leakage control
Feature-importance inspection in the first modelling round showed that nearest-type distances were the most important feature group (permutation drop 0.41 versus 0.11 for the next group). Nearest-type distances are unbounded: such a distance can describe cells beyond the 30 um feature radius, including the target ring. The strict features cap them at 30 um, and the reach guarantee is tested by deleting every cell beyond 30 um and checking that the features do not change. Strict versus original features: 0.746 (95% CI 0.733-0.756) versus 0.833 (95% CI 0.818-0.846) (difference -0.087 (95% CI -0.098 to -0.077)). Non-spatial validation is unsafe: random (non-spatial) cross-validation inflated a coordinate-only model from 0.572 (spatial CV) to 0.858, but changed the strict spatial features only from 0.746 to 0.762 and expression from 0.731 to 0.743.

## 6. Robustness
* Buffer size: 0.745-0.748 across buffers 0-150 um.
* Block size and partition seed: 0.746-0.748 across block sizes [250, 500, 1000] and 3 partition seeds.
* Moving the target ring outward (strict spatial features): ring 50-100 um = 0.746 (own-type 0.612); ring 75-125 um = 0.715 (own-type 0.566); ring 100-150 um = 0.688 (own-type 0.528); the target labels at the shifted rings agree only partly with the primary labels (ARI 0.71 and 0.54), so part of the decline reflects a different target.
* Boundary cells: strict spatial model: 0.767 on high-margin cells vs 0.556 on the 10% lowest-margin cells.

## 7. Limitations
* Single tissue section from one patient: confidence intervals reflect variation across tissue blocks, not across patients; generalisation is untested.
* Cell types and the target are derived from the same 313-gene expression data, so the own-type baseline and the expression-only model are partly circular with the target; the analysis excluding marker genes mitigates but does not remove this.
* Tumour and normal epithelium cannot be separated with this panel without copy-number evidence.
* The target is a gradient cut into three classes (silhouette 0.34 at k = 3); cells near class boundaries are classified much less reliably.
* The T-cell and endothelial labels have the lowest cell-level marker agreement (about 0.6), so results involving them carry extra label uncertainty.
* Hyper-parameters were tuned by nested spatial cross-validation in Notebook 05 and reused in Notebook 06 without tuning on test folds; training cells were sub-sampled for speed.
* An early version of the analysis contained a leakage (unbounded nearest-type distances). It was found by feature-importance inspection, confirmed, and removed; other undetected leakage routes cannot be excluded, which is why the design rules are enforced by tests.

## 8. Reproducibility
Code is in `src/spatialml` (installable, 25 tests, CI); `python -m pytest` and `spatialml demo` run without the data. Package versions used in the original runs are listed in `docs/reproducibility.md`. The notebooks in `notebooks/` reproduce each analysis step on Google Colab.
