# Results

* `figures/` - figures produced by the notebooks (extracted from the executed notebooks; `eval_feature_set_ablation.png` was re-drawn from the table for a legible axis label).
* `tables/` - result tables. **Provenance:** recovered from the tables displayed in the executed notebooks, so they carry display precision (3-4 decimals) and a few are the *displayed* top-N rows rather than the full CSV written to Drive (`feature_class_association_top20.csv`, `feature_qc_top_missingness.csv`, `enrichment_survive_tile_null_top12.csv`, `eval_partition_sensitivity_summary.csv`). Re-running the notebooks regenerates the full files.
* `notebook05_original_features/` - results of Notebook 05, which used unbounded nearest-type distances that reach into the target ring. **Do not quote these as results**; they document the leakage that was found and fixed in Notebook 06 (the corrected results are the `eval_*` files).
