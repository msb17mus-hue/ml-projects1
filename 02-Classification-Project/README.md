# Project 2: Diabetes Classification

Binary classification project that predicts whether a patient has diabetes from eight diagnostic measurements (Pima Indians Diabetes dataset).

## Dataset

768 patient records loaded directly inside the notebook from:

`https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`

The file has no header, so the notebook supplies the column names.

## Workflow

1. Load, inspect and clean the data. Impossible zero values in glucose, blood pressure, skin thickness, insulin and BMI are treated as missing.
2. Exploratory analysis: class balance, distributions by outcome, box plots, correlations.
3. Stratified train/test split first, then a single `Pipeline` with feature engineering, median imputation and robust scaling.
4. Compare Logistic Regression, K-Nearest Neighbors, SVM, Decision Tree, Random Forest and Gradient Boosting with stratified 5-fold cross-validation.
5. Select the best model by cross-validated ROC AUC, then report accuracy, precision, recall, F1, ROC AUC, confusion matrix, ROC curves and permutation importance on the test set.
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

## Disclaimer

Educational project. Predictions are not medical advice.

## Files

| File | Purpose |
|------|---------|
| `notebook.ipynb` | Full machine learning workflow |
| `features.py` | Shared constants and feature engineering used by notebook and app |
| `app.py` | Streamlit application |
| `model.pkl` | Saved pipeline (created by the notebook) |
| `metadata.json` | Input ranges and metrics for the app (created by the notebook) |
| `requirements.txt` | Python dependencies |
