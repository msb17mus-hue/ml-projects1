# Machine Learning Projects Portfolio

A collection of five end-to-end Machine Learning projects covering Regression, Classification, Customer Analytics, Clustering, and Natural Language Processing (NLP).

## Projects

| # | Project | Machine Learning Task | Objective |
|---|---|---|---|
| 01 | House Price Prediction | Supervised Learning – Regression | Predict house prices based on property characteristics. |
| 02 | Classification Project | Supervised Learning – Classification | Train and compare classification algorithms on a real-world dataset. |
| 03 | Customer Churn Prediction | Supervised Learning – Classification | Predict whether customers are likely to leave a service. |
| 04 | Customer Segmentation | Unsupervised Learning – Clustering | Discover customer groups using K-Means clustering. |
| 05 | Sentiment Analysis | Natural Language Processing | Classify text sentiment as positive or negative. |

## Repository Structure

```text
ML-Portfolio/
│
├── 01-House-Price-Prediction/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
│
├── 02-Classification-Project/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
│
├── 03-Customer-Churn-Prediction/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
│
├── 04-Customer-Segmentation/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
│
└── 05-Sentiment-Analysis/
    ├── notebook.ipynb
    ├── app.py
    ├── model.pkl
    ├── requirements.txt
    └── README.md
```

## Project Workflow

Each project follows a complete Machine Learning workflow:

1. Project Introduction
2. Problem Statement
3. Dataset Loading
4. Data Overview
5. Data Cleaning
6. Exploratory Data Analysis (EDA)
7. Data Preprocessing
8. Feature Engineering
9. Model Training
10. Model Comparison
11. Model Evaluation
12. Best Model Selection
13. Model Saving
14. Prediction on New Data
15. Final Conclusion

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle / Joblib

## Installation and Usage

### Clone the Repository

```bash
git clone <repository-url>
cd ML-Portfolio
```

### Install Dependencies

Navigate to the project folder:

```bash
cd 01-House-Price-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

### Run Streamlit Application

```bash
streamlit run app.py
```

Repeat these steps for any of the five projects.

## Model Evaluation

Each project uses suitable evaluation metrics depending on its Machine Learning task.

- Regression: MAE, MSE, RMSE, and R².
- Classification: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- Clustering: Silhouette Score and other clustering evaluation methods.
- Sentiment Analysis: Accuracy, Precision, Recall, and F1-Score.

Actual performance results should be taken from the executed notebooks.

## Key Features

- Real-world Machine Learning problems.
- Public datasets loaded directly through URLs.
- Data cleaning and exploratory analysis.
- Multiple model comparisons.
- Model evaluation and selection.
- Saved trained models.
- Interactive Streamlit applications.
- Reproducible workflows.

## Author

Machine Learning Portfolio

**Purpose:** Demonstrating practical Data Science and Machine Learning skills through end-to-end projects.
