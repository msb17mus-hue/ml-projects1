# Project 3: Customer Churn Prediction

Classification project that predicts whether a telecom customer will cancel the service and identifies the factors that drive churn.

## Dataset

Telco Customer Churn (7,043 customers, 20 columns) loaded directly inside the notebook from:

`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

## Workflow

1. Load, inspect and clean the data (blank `TotalCharges` of new customers, identifier removal, target encoding).
2. Exploratory analysis: class balance, churn rate by contract, internet service, payment method, tenure and more.
3. Stratified train/test split first, then a single `Pipeline` with feature engineering, imputation, scaling and one-hot encoding.
4. Compare Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and Histogram Gradient Boosting with stratified 5-fold cross-validation.
5. Select the best model by cross-validated ROC AUC. Tune the decision threshold for F1 on out-of-fold training predictions, then evaluate once on the test set.
6. Identify churn drivers with permutation importance and logistic regression coefficients.
7. Save the full pipeline to `model.pkl` and app metadata to `metadata.json`.

## Results

Metrics are produced when you run the notebook and are stored in `metadata.json`. The Streamlit app shows them in the **Model performance** and **Churn drivers** tabs.

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
| `metadata.json` | Input options, threshold and metrics for the app (created by the notebook) |
| `requirements.txt` | Python dependencies |
