import pandas as pd

ID_COLUMN = "CustomerID"
COLUMN_RENAMES = {
    "Genre": "Gender",
    "Annual Income (k$)": "AnnualIncome",
    "Spending Score (1-100)": "SpendingScore",
}
CLUSTER_FEATURES = ["Age", "AnnualIncome", "SpendingScore"]
MIN_SEGMENTS = 3
MAX_SEGMENTS = 10


def standardize_columns(frame):
    return frame.rename(columns=COLUMN_RENAMES)


def level(value, median, high_label="High", low_label="Low"):
    return high_label if value >= median else low_label


def describe_segment(centroid, medians):
    return {
        "income": level(centroid["AnnualIncome"], medians["AnnualIncome"]),
        "spending": level(centroid["SpendingScore"], medians["SpendingScore"]),
        "age": level(centroid["Age"], medians["Age"], "Older", "Younger"),
    }
