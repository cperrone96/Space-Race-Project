"""Small-sample, reproducible analysis gate for the IBM SpaceX coursework."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED = {"FlightNumber", "Date", "Launch Site", "Payload Mass (kg)", "Orbit", "Booster Version Category", "class"}


def run(data_path="data/launch_fixture.csv", output_dir="outputs"):
    df = pd.read_csv(data_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    y = df["class"].astype(int)
    prevalence = float(y.mean())
    X = df[["Payload Mass (kg)", "Launch Site", "Orbit", "Booster Version Category"]]
    numeric = ["Payload Mass (kg)"]; categorical = ["Launch Site", "Orbit", "Booster Version Category"]
    pre = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
    model = Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=2000, class_weight=None))])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=32021)
    # Repeated CV creates overlapping test predictions, so cross_val_predict
    # (which requires a partition) is not appropriate. Keep fold metrics to
    # report uncertainty instead.
    fold_metrics = []
    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        yt = y.iloc[test_idx]
        fold_metrics.append({"average_precision": average_precision_score(yt, prob), "roc_auc": roc_auc_score(yt, prob), "brier_score": brier_score_loss(yt, prob)})
    fold = pd.DataFrame(fold_metrics)
    metrics = {
        "n": int(len(df)), "positive_count": int(y.sum()), "class_prevalence": prevalence,
        "majority_baseline_accuracy": float(max(prevalence, 1 - prevalence)),
        "repeated_stratified_cv": {"splits": 5, "repeats": 20, "seed": 32021},
        "average_precision": float(fold.average_precision.mean()),
        "roc_auc": float(fold.roc_auc.mean()),
        "brier_score": float(fold.brier_score.mean()),
        "pr_auc_excess_over_prevalence": float(fold.average_precision.mean() - prevalence),
        "uncertainty_95pct": {k: [float(fold[k].quantile(0.025)), float(fold[k].quantile(0.975))] for k in ["average_precision", "roc_auc", "brier_score"]},
        "evaluation_note": "Repeated stratified cross-validation is uncertainty evidence for this small coursework sample; the original lab's single holdout is not an independent estimate.",
    }
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", default="data/launch_fixture.csv"); p.add_argument("--output-dir", default="outputs"); a = p.parse_args()
    print(json.dumps(run(a.data, a.output_dir), indent=2))
