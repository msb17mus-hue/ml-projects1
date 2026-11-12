import numpy as np
import pandas as pd

TARGET_COLUMN = "Outcome"
RAW_FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ENGINEERED_COLUMNS = ["insulin_missing", "glucose_bmi_index", "pregnancies_per_decade"]
MODEL_COLUMNS = RAW_FEATURE_COLUMNS + ENGINEERED_COLUMNS


def replace_zeros_with_missing(frame):
    cleaned = frame.copy()
    cleaned[ZERO_AS_MISSING_COLUMNS] = cleaned[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    return cleaned


def add_features(frame):
    enriched = replace_zeros_with_missing(frame)
    enriched["insulin_missing"] = enriched["Insulin"].isna().astype(int)
    enriched["glucose_bmi_index"] = enriched["Glucose"] * enriched["BMI"] / 100
    enriched["pregnancies_per_decade"] = enriched["Pregnancies"] / (enriched["Age"] / 10)
    return enriched
