# Limitations

- Single tissue section from one patient: confidence intervals reflect variation across tissue blocks, not across patients; generalisation is untested.
- Cell types and the target are derived from the same 313-gene expression data, so the own-type baseline and the expression-only model are partly circular with the target; the analysis excluding marker genes mitigates but does not remove this.
- Tumour and normal epithelium cannot be separated with this panel without copy-number evidence.
- The target is a gradient cut into three classes (silhouette 0.34 at k = 3); cells near class boundaries are classified much less reliably.
- The T-cell and endothelial labels have the lowest cell-level marker agreement (about 0.6), so results involving them carry extra label uncertainty.
- Hyper-parameters were tuned by nested spatial cross-validation in Notebook 05 and reused in Notebook 06 without tuning on test folds; training cells were sub-sampled for speed.
- An early version of the analysis contained a leakage (unbounded nearest-type distances). It was found by feature-importance inspection, confirmed, and removed; other undetected leakage routes cannot be excluded, which is why the design rules are enforced by tests.
