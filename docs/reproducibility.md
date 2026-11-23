# Reproducibility

## Quick check (2 minutes, no data needed)
```bash
pip install -e ".[dev]"
python -m pytest -q                 # unit and integration tests
spatialml demo --out demo_output    # synthetic end-to-end run
```

## Full analysis (Google Colab, about 3–4 hours in total)
1. `python scripts/download_data.py` (or the download cell in notebook 01).
2. Run the notebooks in `notebooks/` in numeric order, saving to the same project folder (Drive).
3. Long steps (notebooks 05 and 06) write checkpoints; if Colab disconnects, re-run and finished models are loaded from disk.
4. Notebook 07 refactors the code into this package, checks it reproduces the stored features and targets, and writes the report.

## Determinism
* One seed (`configs/default.json`, `seed`) controls sub-sampling, k-means, bootstrap and permutation draws.
* Gradient boosting and k-means are deterministic for a fixed seed and library versions; the random forest uses `n_jobs=-1` but a fixed `random_state`.
* Versions observed in the original Colab runs: Python 3.13, pandas 2.2.3, anndata 0.13.4. Running notebook 07 writes `requirements-lock.txt` and `environment.json` with the complete list for your session.

## Known environment issues
* **Do not downgrade pandas inside Colab notebooks** (e.g. `pip install pandas==2.2.3`). Files written by a newer pandas/anndata cannot always be read by an older one
  (`TypeError: StringDtype.__init__() got an unexpected keyword argument 'na_value'`). The expression loader in notebook 06 reads the HDF5 layer directly to avoid this.
* Colab warns that `google-colab` pins pandas 2.2.x. The package declares support for pandas >= 2.0; it was developed against pandas 3.0, and the test suite (run in notebook 07 and in CI) checks whichever version is installed - read its result rather than assuming.
