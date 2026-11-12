# Project 5: Sentiment Analysis of Movie Reviews

Natural language processing project that classifies movie reviews as positive or negative and shows which words drive each prediction.

## Dataset

IMDB Movie Reviews (50,000 labelled reviews, two balanced classes) loaded directly inside the notebook from a public GitHub raw URL (see `DATA_URLS` in the notebook). The file is about 60 MB.

## Workflow

1. Load, inspect and clean the data (missing values, exact duplicate reviews, label encoding).
2. Exploratory analysis: class balance, review length, most frequent words per class.
3. Stratified train/test split first. Text cleaning and TF-IDF (words and word pairs) are part of one scikit-learn `Pipeline`, so the vocabulary is learned from training data only.
4. Compare Logistic Regression, Multinomial Naive Bayes and a calibrated Linear SVM with stratified 5-fold cross-validation.
5. Select the best model by cross-validated ROC AUC, evaluate once on the test set, and inspect errors and accuracy by review length.
6. Interpret the model with logistic regression term weights.
7. Save the full pipeline to `model.pkl` and app metadata to `metadata.json`.

## Results

Metrics are produced when you run the notebook and are stored in `metadata.json`. The Streamlit app shows them in the **Model performance** and **Vocabulary** tabs.

## Run locally

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
streamlit run app.py
```

Run the notebook first: it creates `model.pkl` and `metadata.json`, which the app needs.

The full dataset takes several minutes to train on. To run faster, set `MAX_ROWS = 10000` in the imports cell of the notebook.

## Deploy

Commit `model.pkl` and `metadata.json` after running the notebook, then deploy `app.py` on Streamlit Community Cloud. Check the size of `model.pkl` first. GitHub rejects files above 100 MB, and a smaller `max_features` value in the notebook reduces the file size.

## Limitations

The model is trained on English movie reviews and may be unreliable on other domains. Sarcasm and mixed reviews are hard for bag-of-words models.

## Files

| File | Purpose |
|------|---------|
| `notebook.ipynb` | Full machine learning workflow |
| `features.py` | Text cleaning, tokenisation and term lookup shared by notebook and app |
| `app.py` | Streamlit application |
| `model.pkl` | Saved pipeline (created by the notebook) |
| `metadata.json` | Metrics, curves and term weights for the app (created by the notebook) |
| `requirements.txt` | Python dependencies |
