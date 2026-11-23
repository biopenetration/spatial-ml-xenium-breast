# Methods

## Question
Does the local cellular neighbourhood of a cell (cells within 30 µm) predict the composition class of the surrounding tissue
(a 50–100 µm ring), beyond what the cell's own transcriptome or cell type tells us?

## Design rules (enforced in code and tests)
1. **Feature reach.** Every feature depends only on cells within `cap_um` (30 µm) of the focal cell; the focal cell is excluded.
   Nearest-type distances are *capped* at 30 µm, because an unbounded distance encodes cells in the target ring.
   `spatialml.features.reach_test` deletes all cells beyond the cap and checks that the features do not change.
2. **Disjoint target.** The target is computed from cells at 50–100 µm only (`spatialml.targets`).
3. **Spatial evaluation.** Tissue is cut into 500 µm blocks assigned to folds. Training cells within 100 µm (the target ring's outer
   radius) of any test cell are removed. Preprocessing is fitted inside each training fold.
4. **Block-level uncertainty.** Confidence intervals resample spatial blocks, not cells. Model comparisons are paired.
5. **Spatial null models.** Feature–class association is compared with a toroidal-shift null that preserves spatial autocorrelation;
   neighbourhood enrichment uses a global and a within-tile (250 µm) label permutation.

## Pipeline
| Step | Code | Notebook |
|---|---|---|
| Data validation and QC | – | 01, 02 |
| Marker-based annotation with uncertainty, label refinement | `spatialml.annotation` | 02, 03 |
| Spatial features | `spatialml.features`, `spatialml.neighborhood` | 03, 06 |
| Target (annulus composition, k-means, margin) | `spatialml.targets` | 03, 04 |
| Statistics and null models | `spatialml.stats` | 04 |
| Models and spatial cross-validation | `spatialml.modeling`, `spatialml.evaluation` | 05, 06 |
| Leakage audit and robustness | `spatialml.features.reach_test`, `spatialml.evaluation` | 06 |

## Leakage found and corrected during the project
An early feature set contained unbounded nearest-type distances. They reached into the target ring and inflated balanced accuracy.
They were replaced by distances capped at 30 µm; the effect is quantified in the report (`reports/scientific_report.md`).
The early results are kept in `results/notebook05_original_features/` for transparency and are **not** the reported results.
