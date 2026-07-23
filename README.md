# Predicting SpaceX Falcon 9 First-Stage Landings

End-to-end data science capstone that predicts whether the **Falcon 9 first
stage will land successfully** — the key driver of SpaceX's launch cost
advantage. The project runs the full pipeline: data collection, cleaning,
exploratory analysis, interactive visualization, and machine-learning
classification.

## Pipeline

| Stage | Notebook | Tools |
|---|---|---|
| Data collection (API) | `jupyter-labs-spacex-data-collection-api.ipynb` | SpaceX REST API, `requests`, pandas |
| Data collection (web scraping) | `SpaceX Webscraping.ipynb` | `BeautifulSoup`, Wikipedia |
| Data wrangling | `SpaceX Data Wrangling.ipynb` | pandas, feature/label engineering |
| EDA with SQL | `SpaceX EDA with SQL.ipynb` | SQLite, `ipython-sql` |
| EDA & feature engineering | `SpaceX EDA and Ft Engineering.ipynb` | pandas, matplotlib, seaborn, one-hot encoding |
| Interactive maps | `SpaceX Interactive Visual Analytics with Folium.ipynb` | Folium |
| Interactive dashboard | `spacex_dash_app (1).py` | Plotly Dash |
| Predictive modeling | `SpaceX Prediction Analysis.ipynb` | scikit-learn (LogReg, SVM, Decision Tree, KNN), GridSearchCV |

## Approach

1. **Collect** launch records from the SpaceX API and by scraping Wikipedia.
2. **Wrangle** into a clean dataset and derive a binary landing-outcome label.
3. **Explore** launch-site, payload, orbit, and booster-version relationships
   with SQL and visual EDA.
4. **Visualize** launch sites and outcomes interactively (Folium + a Plotly
   Dash app).
5. **Model** landing success with several classifiers, tuning hyperparameters
   via `GridSearchCV` and comparing accuracy on a held-out test set.

## Skills demonstrated

REST API extraction · web scraping · data wrangling · SQL EDA · interactive
geospatial viz · dashboarding · supervised classification & model selection.

## Run it

```bash
pip install pandas numpy requests beautifulsoup4 matplotlib seaborn scikit-learn folium plotly dash
jupyter lab
```

Run the notebooks in the pipeline order above. To launch the dashboard:
`python "spacex_dash_app (1).py"` then open the local URL it prints.

---
*IBM Data Science Professional Certificate capstone.*
