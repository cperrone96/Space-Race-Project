# Falcon 9 first-stage landing prediction — IBM coursework

This repository is explicitly an **IBM Data Science Professional Certificate
capstone coursework artifact**. It documents completed course labs (API
collection, web scraping, SQL, EDA, Folium, Dash, and classification). It is
not professional SpaceX work, production experience, or a deployed launch
decision system.

## Reproducible local gate

The original IBM static extracts are remote course materials. Because they are
not bundled, `data/launch_fixture.csv` is a checked-in schema/smoke fixture: it
preserves the 90-row course label vector and deterministic feature columns but
must not be interpreted as newly collected SpaceX observations. URLs, source
versions, and the fixture SHA-256 are recorded in
[`data/SOURCE_MANIFEST.json`](data/SOURCE_MANIFEST.json).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python spacex_analysis.py --data data/launch_fixture.csv --output-dir outputs
pytest -q
```

The analysis reports class prevalence, a majority-class accuracy baseline,
average precision (PR-AUC), ROC-AUC, Brier score, and repeated stratified
5-fold cross-validation (20 repeats). The small sample makes uncertainty wide;
the original single holdout and hyperparameter-search scores should not be
treated as independent performance estimates. The output is a coursework
demonstration, not evidence that a model can guide launch or safety decisions.

## Results and limitations

The current local fixture has 90 rows and 66.7% positive labels, so the
majority accuracy baseline is 66.7%. Run the command above for the exact
cross-validated PR metrics in `outputs/metrics.json`. These metrics describe
the fixture only. The feature set is a retrospective teaching exercise, the
label prevalence is small-sample dependent, and there is no prospective or
time-based validation. No causal or operational conclusions are claimed.

## Coursework notebooks and dashboard

The notebooks retain the IBM lab sequence and are archival educational
materials. The canonical local smoke path above is network-independent. The
Dash app reads `data/launch_fixture.csv` by default (or `SPACEX_DATA_PATH`) and
can be launched with `python "spacex_dash_app (1).py"`; it is a visualization
demo, not a production dashboard. The historical web-scraping notebook is kept
separate from the canonical fixture path because live scraping is inherently
fragile and source pages change.
