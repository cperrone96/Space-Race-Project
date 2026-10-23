"""Invariants for the Falcon 9 landing analysis. Runs in a few seconds."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import spacex_analysis as sa  # noqa: E402


def test_raw_files_match_manifest():
    manifest = json.loads((ROOT / "data" / "SOURCE_MANIFEST.json").read_text())
    for rel, meta in manifest["files"].items():
        assert hashlib.sha256((ROOT / "data" / rel).read_bytes()).hexdigest() == meta["sha256"], rel


def test_launch_table_shape_and_label():
    data = sa.load_launches()
    assert len(data) == 90 and data["Class"].sum() == 60
    assert data["FlightNumber"].is_monotonic_increasing
    # The course's full-data mean fill is undone so imputation happens inside training folds.
    assert data["PayloadMass"].isna().sum() == 5


def test_prelaunch_features_exclude_post_launch_columns():
    data = sa.load_launches()
    course = sa.pd.read_csv(sa.RAW / "dataset_part_3.csv")
    X, _, _ = sa.feature_sets(data, course)["prelaunch"]
    assert not set(sa.LEAKY_COLUMNS) & set(X.columns)
    assert "Class" not in X.columns and "Outcome" not in X.columns


def test_reused_count_reveals_label_for_boosters_that_fly_again():
    audit = sa.leakage_audit(sa.load_launches())
    assert audit["launches_with_reusedcount_gt0_and_booster_flies_again"] == audit["of_which_landed"] > 0


def test_one_step_ahead_never_sees_the_future():
    data = sa.load_launches()
    X, num, cat = sa.feature_sets(data, sa.pd.read_csv(sa.RAW / "dataset_part_3.csv"))["prelaunch"]
    y = data["Class"].to_numpy()
    model, grid = sa.candidate_models()["logistic_regression"]
    start = 80
    base = sa.one_step_ahead_probabilities(X, y, model, grid, num, cat, start)
    flipped = y.copy()
    flipped[-1] = 1 - flipped[-1]  # the last flight is never in any training window
    assert np.allclose(base, sa.one_step_ahead_probabilities(X, flipped, model, grid, num, cat, start))


def test_nested_cv_beats_majority_baseline(monkeypatch):
    monkeypatch.setattr(sa, "N_REPEATS", 2)
    data = sa.load_launches()
    X, num, cat = sa.feature_sets(data, sa.pd.read_csv(sa.RAW / "dataset_part_3.csv"))["prelaunch"]
    y = data["Class"].to_numpy()
    lr = sa.nested_cv(X, y, *sa.candidate_models()["logistic_regression"], num, cat)
    base = sa.nested_cv(X, y, *sa.candidate_models()["majority_baseline"], num, cat)
    assert all(r["roc_auc"] == pytest.approx(0.5) for r in base)
    assert np.mean([r["roc_auc"] for r in lr]) > 0.7
    assert all(0 <= r["brier"] <= 1 for r in lr)
