# Project 4: Customer Segmentation

Unsupervised learning project that groups mall customers into behavioural segments with clustering.

## Dataset

Mall Customers (200 customers: age, gender, annual income, spending score) loaded directly inside the notebook from a public GitHub raw URL (see `DATA_URLS` in the notebook).

## Workflow

1. Load, inspect and clean the data. `Gender` is kept only for profiling, not for clustering.
2. Exploratory analysis: distributions, pairwise scatter plots, correlations.
3. Standardise the features and project them with PCA for visualisation.
4. Choose the number of clusters with inertia, silhouette, Davies-Bouldin and Calinski-Harabasz.
5. Compare K-Means, Agglomerative clustering and a Gaussian Mixture Model.
6. Check stability with the adjusted Rand index across random starts and subsamples.
7. Fit the final K-Means pipeline, profile each segment with data-driven labels and save `model.pkl` and `metadata.json`.

## Results

There is no target variable, so no accuracy is reported. Cluster-quality metrics and segment profiles are produced when you run the notebook and are stored in `metadata.json`. The Streamlit app shows them in the **Segment explorer** and **Model quality** tabs.

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
| `features.py` | Shared constants and helper functions used by notebook and app |
| `app.py` | Streamlit application |
| `model.pkl` | Saved scaler and K-Means pipeline (created by the notebook) |
| `metadata.json` | Segment profiles and quality metrics for the app (created by the notebook) |
| `requirements.txt` | Python dependencies |
