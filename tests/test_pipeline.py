import json

from spatialml import pipeline


def test_demo_pipeline_runs_and_beats_chance(tmp_path):
    s = pipeline.run_demo(tmp_path, n_cells=3000, seed=0, n_bootstrap=50)
    assert (tmp_path / "demo_results.csv").exists() and (tmp_path / "demo_summary.json").exists()
    r = s["results"]
    assert all(0.0 <= v <= 1.0 for v in r.values())
    assert r["spatial_strict"] > s["chance"] + 0.1                       # real signal exists in the synthetic tissue
    assert r["coordinates_random_cv"] >= r["coordinates_spatial_cv"] - 0.02     # random CV is never harder than spatial CV
    assert json.load(open(tmp_path / "demo_summary.json"))["config"]["features"]["cap_um"] == 30
