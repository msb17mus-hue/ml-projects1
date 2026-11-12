import pandas as pd

TARGET_COLUMN = "median_house_value"
CATEGORICAL_COLUMNS = ["ocean_proximity"]
RAW_NUMERIC_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
RAW_FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
ENGINEERED_COLUMNS = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS
MAX_ROOMS_PER_HOUSEHOLD = 20
MAX_PEOPLE_PER_HOUSEHOLD = 20


def add_features(frame):
    enriched = frame.copy()
    enriched["rooms_per_household"] = enriched["total_rooms"] / enriched["households"]
    enriched["bedrooms_per_room"] = enriched["total_bedrooms"] / enriched["total_rooms"]
    enriched["population_per_household"] = enriched["population"] / enriched["households"]
    return enriched
