"""Falcon 9 first-stage landing prediction with leakage-controlled evaluation.

The IBM capstone feature table (``dataset_part_3.csv``) includes two columns
built from information that did not exist at launch time:

* ``ReusedCount`` is the booster core's *lifetime* reuse count, read from the
  SpaceX API when the data was collected, not the count before this flight.
* ``Serial`` one-hot columns identify the booster, so a booster that flew again
  later (and therefore must have been recovered) is recognizable.

This module compares models on the course feature set and on a pre-launch
feature set with those columns removed. Every model is tuned with an inner
cross-validation loop inside each outer training fold (nested CV), and a
one-step-ahead evaluation trains only on earlier flights.

Run ``python spacex_analysis.py`` to regenerate ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
SEED = 2021
N_REPEATS = 10
N_OUTER = 5
N_INNER = 5

TARGET = "Class"
# Known before launch: mission profile and booster configuration.
PRELAUNCH_NUMERIC = ["FlightNumber", "PayloadMass", "Flights", "Block"]
PRELAUNCH_BINARY = ["GridFins", "Reused", "Legs"]
PRELAUNCH_CATEGORICAL = ["Orbit", "LaunchSite", "LandingPad"]
# Built from post-launch information (see module docstring).
LEAKY_COLUMNS = ["ReusedCount", "Serial"]
# The course's fill value for missing payload mass: the mean of all launches.
COURSE_IMPUTED_PAYLOAD_KG = 6104.959411764706


def load_launches(path: Path = RAW / "dataset_part_2.csv") -> pd.DataFrame:
    """Load the 90 Falcon 9 launches and undo the course's full-data mean imputation.

    The course filled missing payload masses with the mean of all launches
    before any train/test split. Those rows are set back to missing so the
    median is learned inside each training fold.
    """
    data = pd.read_csv(path)
    data.loc[np.isclose(data["PayloadMass"], COURSE_IMPUTED_PAYLOAD_KG), "PayloadMass"] = np.nan
    data["LandingPad"] = data["LandingPad"].fillna("none")
    for column in PRELAUNCH_BINARY:
        data[column] = data[column].astype(int)
    return data.sort_values("FlightNumber").reset_index(drop=True)


def feature_sets(data: pd.DataFrame, course_features: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, list[str], list[str]]]:
    """Return {name: (X, numeric_columns, categorical_columns)}."""
    prelaunch = data[PRELAUNCH_NUMERIC + PRELAUNCH_BINARY + PRELAUNCH_CATEGORICAL]
    course = course_features.astype(float)
    return {
        "prelaunch": (prelaunch, PRELAUNCH_NUMERIC + PRELAUNCH_BINARY, PRELAUNCH_CATEGORICAL),
        "course_with_leaky_columns": (course, list(course.columns), []),
    }


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    steps = [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric)]
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(steps)


def candidate_models() -> dict[str, tuple[object, dict[str, list]]]:
    """The four course classifiers with compact grids, plus a majority baseline."""
    return {
        "majority_baseline": (DummyClassifier(strategy="prior"), {}),
        "logistic_regression": (LogisticRegression(max_iter=5000), {"model__C": [0.01, 0.1, 1.0, 10.0]}),
        "svm_rbf": (SVC(kernel="rbf", probability=True, random_state=SEED), {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.01, 0.1]}),
        "decision_tree": (DecisionTreeClassifier(random_state=SEED), {"model__max_depth": [2, 3, 4, 6], "model__min_samples_leaf": [1, 3, 5]}),
        "knn": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 7, 9]}),
    }


def build_estimator(model, grid: dict, numeric: list[str], categorical: list[str]):
    pipe = Pipeline([("prep", preprocessor(numeric, categorical)), ("model", clone(model))])
    if not grid:
        return pipe
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1)


def score(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, prob)),
    }


def nested_cv(X: pd.DataFrame, y: np.ndarray, model, grid, numeric, categorical) -> list[dict[str, float]]:
    """Repeated stratified 5-fold CV; each repeat pools out-of-fold predictions for all rows."""
    per_repeat = []
    for repeat in range(N_REPEATS):
        outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED + repeat)
        prob = np.empty(len(y))
        for train, test in outer.split(X, y):
            est = build_estimator(model, grid, numeric, categorical).fit(X.iloc[train], y[train])
            prob[test] = est.predict_proba(X.iloc[test])[:, 1]
        per_repeat.append(score(y, prob))
    return per_repeat


def one_step_ahead_probabilities(X: pd.DataFrame, y: np.ndarray, model, grid, numeric, categorical, start: int = 40) -> np.ndarray:
    """Train on flights 1..t and predict flight t+1, for t = start..n-1. Rows must be in flight order."""
    prob = []
    for t in range(start, len(y)):
        est = build_estimator(model, grid, numeric, categorical).fit(X.iloc[:t], y[:t])
        prob.append(est.predict_proba(X.iloc[[t]])[0, 1])
    return np.asarray(prob)


def one_step_ahead(X: pd.DataFrame, y: np.ndarray, model, grid, numeric, categorical, start: int = 40) -> dict[str, float]:
    """Time-respecting evaluation: score the one-step-ahead predictions."""
    prob = one_step_ahead_probabilities(X, y, model, grid, numeric, categorical, start)
    out = score(y[start:], prob)
    out["n_predictions"] = len(prob)
    out["positive_rate"] = float(y[start:].mean())
    return out


def summarize(per_repeat: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(per_repeat)
    return {m: {"mean": float(frame[m].mean()), "min": float(frame[m].min()), "max": float(frame[m].max())} for m in frame}


def leakage_audit(data: pd.DataFrame) -> dict[str, int]:
    """Show that ReusedCount encodes the outcome for boosters that fly again later."""
    flies_again = data.duplicated("Serial", keep="last")  # rows are in flight order
    reused_and_flies_again = (data["ReusedCount"] > 0) & flies_again
    return {
        "launches_with_reusedcount_gt0_and_booster_flies_again": int(reused_and_flies_again.sum()),
        "of_which_landed": int(data.loc[reused_and_flies_again, TARGET].sum()),
        "payload_mass_rows_mean_imputed_by_course": int(data["PayloadMass"].isna().sum()),
    }


def run(output_dir: Path = ROOT / "outputs") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_launches()
    course = pd.read_csv(RAW / "dataset_part_3.csv")
    if not np.array_equal(course["FlightNumber"].to_numpy(), data["FlightNumber"].to_numpy()):
        raise ValueError("feature table and launch table are misaligned")
    y = data[TARGET].to_numpy()

    rows, one_step = [], {}
    for set_name, (X, numeric, categorical) in feature_sets(data, course).items():
        for model_name, (model, grid) in candidate_models().items():
            summary = summarize(nested_cv(X, y, model, grid, numeric, categorical))
            for metric, stats in summary.items():
                rows.append({"feature_set": set_name, "model": model_name, "metric": metric, **stats})
            if set_name == "prelaunch":
                one_step[model_name] = one_step_ahead(X, y, model, grid, numeric, categorical)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "model_comparison.csv", index=False)
    auc = table[table.metric.eq("roc_auc")].set_index(["feature_set", "model"])["mean"]
    best_prelaunch = auc.loc["prelaunch"].drop("majority_baseline").idxmax()

    metrics = {
        "n_launches": int(len(data)),
        "landed": int(y.sum()),
        "landing_rate": float(y.mean()),
        "validation": {
            "nested_cv": f"{N_REPEATS} repeats x {N_OUTER}-fold stratified outer CV, {N_INNER}-fold inner grid search on ROC-AUC",
            "one_step_ahead": "train on flights 1..t, predict flight t+1, t = 40..89",
            "seed": SEED,
        },
        "leakage_audit": leakage_audit(data),
        "best_prelaunch_model": best_prelaunch,
        "nested_cv_roc_auc": {
            fs: {m: summarize_row(table, fs, m, "roc_auc") for m in candidate_models()} for fs in ["prelaunch", "course_with_leaky_columns"]
        },
        "nested_cv_balanced_accuracy_prelaunch": {m: summarize_row(table, "prelaunch", m, "balanced_accuracy") for m in candidate_models()},
        "one_step_ahead_prelaunch": one_step,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_model_comparison(metrics, output_dir / "figures" / "model_comparison.png")
    return metrics


MODEL_LABELS = {
    "majority_baseline": "Majority baseline",
    "logistic_regression": "Logistic regression",
    "svm_rbf": "SVM (RBF)",
    "decision_tree": "Decision tree",
    "knn": "k-nearest neighbors",
}


def plot_model_comparison(metrics: dict, path: Path) -> None:
    """Dot plot: random nested-CV ROC-AUC (with range over repeats) vs one-step-ahead ROC-AUC."""
    from matplotlib.figure import Figure  # object API: no global backend or pyplot state

    ink, muted, blue, orange, surface = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#fcfcfb"
    models = list(MODEL_LABELS)
    cv = metrics["nested_cv_roc_auc"]["prelaunch"]
    ahead = metrics["one_step_ahead_prelaunch"]
    fig = Figure(figsize=(8, 4.2), dpi=150, facecolor=surface)
    ax = fig.subplots()
    ax.set_facecolor(surface)
    for i, model in enumerate(models):
        y = len(models) - 1 - i
        ax.plot([cv[model]["min"], cv[model]["max"]], [y + 0.12] * 2, color=blue, lw=2, solid_capstyle="round")
        ax.plot(cv[model]["mean"], y + 0.12, "o", ms=8, color=blue, mec=surface, mew=2,
                label="Shuffled nested CV (mean, range over 10 repeats)" if i == 0 else None)
        ax.plot(ahead[model]["roc_auc"], y - 0.12, "D", ms=7, color=orange, mec=surface, mew=2,
                label=f"Time-ordered, one step ahead (n = {ahead[model]['n_predictions']} flights)" if i == 0 else None)
    ax.axvline(0.5, color=muted, lw=1, ls=":")
    ax.text(0.505, len(models) - 0.55, "chance", color=muted, fontsize=8, va="bottom")
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in reversed(models)], color=ink)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.5, len(models) - 0.3)
    ax.set_xlabel("ROC-AUC (pre-launch features)", color=muted)
    ax.tick_params(colors=muted, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")
    ax.grid(axis="x", color="#e9e8e4", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, fontsize=8, labelcolor=ink)
    ax.set_title("Shuffled CV looks strong; predicting the next flight is much harder", loc="left", fontsize=11, color=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)


def summarize_row(table: pd.DataFrame, feature_set: str, model: str, metric: str) -> dict[str, float]:
    row = table[(table.feature_set == feature_set) & (table.model == model) & (table.metric == metric)].iloc[0]
    return {"mean": round(float(row["mean"]), 4), "min": round(float(row["min"]), 4), "max": round(float(row["max"]), 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
