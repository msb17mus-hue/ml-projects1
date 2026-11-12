# Project 1: House Price Prediction

Regression project that predicts the median house value of a California housing district.

## Dataset

California housing census data (20,640 districts) loaded directly inside the notebook from:

`https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv`

## Workflow

1. Load data from the URL, inspect, clean (duplicates, censored target values, implausible districts).
2. Exploratory data analysis (distributions, correlations, geography, categorical effects).
3. Train/test split first, then a single scikit-learn `Pipeline` with feature engineering, imputation, scaling and one-hot encoding.
4. Compare Linear Regression, Random Forest and Gradient Boosting with 5-fold cross-validation on the training set.
5. Select the best model by cross-validated R², evaluate once on the test set (MAE, RMSE, R²) and compute permutation importance.
6. Save the full pipeline to `model.pkl` and app metadata to `metadata.json`.

## Results

Metrics are produced when you run the notebook and are stored in `metadata.json`. The Streamlit app shows them in the **Model performance** tab.

## Run locally

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
streamlit run app.py
```

Run the notebook first: it creates `model.pkl` and `metadata.json`, which the app needs.

## Deploy

Commit `model.pkl` and `metadata.json` after running the notebook, then deploy `app.py` on Streamlit Community Cloud.

## Files

| File | Purpose |
|------|---------|
| `notebook.ipynb` | Full machine learning workflow |
| `features.py` | Shared constants and feature engineering used by notebook and app |
| `app.py` | Streamlit application |
| `model.pkl` | Saved pipeline (created by the notebook) |
| `metadata.json` | Input ranges and metrics for the app (created by the notebook) |
| `requirements.txt` | Python dependencies |
