# Notebooks (run in order, Google Colab)

| File | Module | Purpose |
|---|---|---|
| `01_data_engineering_and_qc.ipynb` | 1 | download, validate cell IDs and coordinates, QC |
| `02_annotation_and_target.ipynb` | 2 | clustering, marker-based labels with uncertainty, stability checks |
| `03_spatial_features_and_target.ipynb` | 3 | label refinement, neighbourhood features, microenvironment target, spatial folds |
| `04_statistical_testing.ipynb` | 4 | effect sizes, block bootstrap, spatial null models, neighbourhood enrichment |
| `05_ml_baselines_original_features.ipynb` | 5 | model ladder under spatial CV. **Uses the original feature set that contained long-range distances (leakage); kept for transparency, not for reporting.** |
| `06_leakage_and_robust_evaluation.ipynb` | 6 | reach audit, strict features, expression vs spatial, sensitivity analyses |
| `07_reproducible_software_and_portfolio.ipynb` | 7 | builds this package, tests, and the report |

Outputs are cleared so the files stay small and diff-friendly; the figures and tables they produced are in `results/`.
Notebooks mount Google Drive and read/write a project folder (`spatial_ml_breast_cancer`); change `PROJECT_DIR` at the top if you use another location.
