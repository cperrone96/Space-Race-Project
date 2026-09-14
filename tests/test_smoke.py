import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from spacex_analysis import run  # noqa: E402


def test_fixture_schema_and_checksum_manifest():
    manifest = json.loads((ROOT / "data" / "SOURCE_MANIFEST.json").read_text())
    data = ROOT / "data" / manifest["local_fixture"]
    import hashlib
    assert hashlib.sha256(data.read_bytes()).hexdigest() == manifest["local_fixture_sha256"]
    frame = pd.read_csv(data)
    assert len(frame) == 90
    assert frame["class"].isin([0, 1]).all()


def test_analysis_has_prevalence_baseline_and_pr_metrics(tmp_path):
    metrics = run(ROOT / "data" / "launch_fixture.csv", tmp_path)
    assert metrics["n"] == 90
    assert metrics["majority_baseline_accuracy"] >= metrics["class_prevalence"]
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
