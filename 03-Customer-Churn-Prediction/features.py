import pandas as pd

TARGET_COLUMN = "Churn"
RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
RAW_NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [column for column in RAW_FEATURE_COLUMNS if column not in RAW_NUMERIC_COLUMNS]
SERVICE_COLUMNS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
ENGINEERED_COLUMNS = ["services_count", "is_new_customer", "has_family"]
NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS
NEW_CUSTOMER_MONTHS = 12


def add_features(frame):
    enriched = frame.copy()
    enriched["services_count"] = (enriched[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    enriched["is_new_customer"] = (enriched["tenure"] <= NEW_CUSTOMER_MONTHS).astype(int)
    enriched["has_family"] = ((enriched["Partner"] == "Yes") | (enriched["Dependents"] == "Yes")).astype(int)
    return enriched
