"""Smoke test: train.py, compare.py and recalibrate.py run end to end on synthetic data. Run with `pytest ml`."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args):
    r = subprocess.run([sys.executable, "-W", "ignore", *args], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_pipeline_runs_on_synthetic_data(tmp_path):
    run("make_synthetic.py", "--n", "1500", "--out", str(tmp_path / "s.csv"))
    run("make_synthetic.py", "--n", "400", "--ntt", "--seed", "2", "--out", str(tmp_path / "ntt.csv"))
    out = run("train.py", "--data", str(tmp_path / "s.csv"), "--holdout-province", "52", "--weight-col", "weight",
              "--bootstrap", "20", "--out", str(tmp_path / "out"))
    assert "Thesis model" in out and (tmp_path / "out" / "model.joblib").exists()
    out = run("compare.py", "--data", str(tmp_path / "s.csv"), "--base-drop-extras", "--reps", "20")
    assert "new LR + extra inputs" in out
    run("recalibrate.py", "--local", str(tmp_path / "ntt.csv"), "--capacity", "0.3", "--bootstrap", "20", "--out", str(tmp_path / "ntt"))
    new = json.loads((tmp_path / "ntt" / "nutrisense_risk_model.json").read_text())
    old = json.loads((HERE.parent / "backend/app/ai/riskModel/nutrisense_risk_model.json").read_text())
    assert new["coef"] == old["coef"] and new["intercept"] == old["intercept"]  # only the threshold changes
    assert new["version"].endswith("-ntt")
