# Evaluation report

## Design
- Target: composition class of the 50-100 um ring; features: cells within 30 um (self excluded, distances capped at 30 um).
- Evaluation: 5 spatial block folds (500 um blocks, 100 um buffer); block-bootstrap CIs; paired comparisons.

## Results
1. Leakage control: removing long-range distance information changes balanced accuracy from 0.833 (unbounded distances) to 0.746 (reach <= 30 um); difference -0.087 (95% CI -0.098 to -0.077).
2. Beyond own cell type: strict spatial features reach 0.746 (CI 0.733-0.756) vs 0.612 for the cross-validated own-type baseline; difference +0.133 (95% CI +0.124 to +0.143).
3. Expression alone (own transcriptome): 0.731; strict spatial alone: 0.746; spatial minus expression +0.014 (95% CI +0.004 to +0.024).
4. Main research question - spatial features beyond expression: combined minus expression-only +0.042 (95% CI +0.035 to +0.049); combined minus spatial-only +0.027 (95% CI +0.022 to +0.033).
5. Expression without the marker genes used to define cell types still gives 0.711 (vs own-type baseline +0.099 (95% CI +0.088 to +0.110)).
6. Random CV inflates performance: coordinates-only 0.858 (random) vs 0.572 (spatial); strict spatial 0.762 vs 0.746.
7. Buffer sensitivity: balanced accuracy ranges 0.745-0.748 across buffers 0-150 um.
8. Partition sensitivity: 0.746-0.748 across block sizes [250, 500, 1000] and 3 partition seeds.
9. Target-ring sensitivity (strict spatial): ring 50-100 um = 0.746 (own-type 0.612); ring 75-125 um = 0.715 (own-type 0.566); ring 100-150 um = 0.688 (own-type 0.528).
10. Boundary cells: strict spatial model 0.767 on high-margin cells vs 0.556 on the 10% lowest-margin cells.

## Limitations
- Single tissue section from one patient: confidence intervals reflect variation across tissue blocks, not across patients; generalisation is untested.
- Cell types and the target are derived from the same 313-gene expression data, so the own-type baseline and the expression-only model are partly circular with the target; the analysis excluding marker genes mitigates but does not remove this.
- Tumour and normal epithelium cannot be separated with this panel without copy-number evidence.
- The target is a gradient cut into three classes (silhouette 0.34 at k = 3); cells near class boundaries are classified much less reliably.
- The T-cell and endothelial labels have the lowest cell-level marker agreement (about 0.6), so results involving them carry extra label uncertainty.
- Hyper-parameters were tuned by nested spatial cross-validation in Notebook 05 and reused in Notebook 06 without tuning on test folds; training cells were sub-sampled for speed.
- An early version of the analysis contained a leakage (unbounded nearest-type distances). It was found by feature-importance inspection, confirmed, and removed; other undetected leakage routes cannot be excluded, which is why the design rules are enforced by tests.
