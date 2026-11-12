import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from features import (
    MAX_PEOPLE_PER_HOUSEHOLD,
    MAX_ROOMS_PER_HOUSEHOLD,
    RAW_FEATURE_COLUMNS,
)

BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIRECTORY / "model.pkl"
METADATA_PATH = BASE_DIRECTORY / "metadata.json"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def validate_inputs(values):
    problems = []
    if values["total_bedrooms"] is not None and values["total_bedrooms"] > values["total_rooms"]:
        problems.append("Total bedrooms cannot exceed total rooms.")
    if values["households"] > values["population"] * 2:
        problems.append("The number of households is too high compared with the population.")
    if values["total_rooms"] / values["households"] > MAX_ROOMS_PER_HOUSEHOLD:
        problems.append(f"More than {MAX_ROOMS_PER_HOUSEHOLD} rooms per household is outside the training range.")
    if values["population"] / values["households"] > MAX_PEOPLE_PER_HOUSEHOLD:
        problems.append(f"More than {MAX_PEOPLE_PER_HOUSEHOLD} people per household is outside the training range.")
    return problems


def numeric_field(label, column, ranges, step, help_text=None, disabled=False):
    limits = ranges[column]
    return st.number_input(
        label,
        min_value=float(limits["min"]),
        max_value=float(limits["max"]),
        value=float(limits["median"]),
        step=step,
        help=help_text,
        disabled=disabled,
    )


def collect_inputs(metadata):
    ranges = metadata["numeric_ranges"]
    categories = metadata["categories"]["ocean_proximity"]
    left, right = st.columns(2)
    with left:
        st.subheader("Location")
        longitude = numeric_field("Longitude", "longitude", ranges, 0.01)
        latitude = numeric_field("Latitude", "latitude", ranges, 0.01)
        ocean_proximity = st.selectbox("Ocean proximity", categories)
        st.subheader("Neighbourhood")
        median_income = numeric_field(
            "Median income (tens of thousands USD)", "median_income", ranges, 0.1,
            "A value of 3.5 means about 35,000 USD per year.",
        )
        population = numeric_field("Population", "population", ranges, 10.0)
        households = numeric_field("Households", "households", ranges, 5.0)
    with right:
        st.subheader("Housing stock")
        housing_median_age = numeric_field("Median house age (years)", "housing_median_age", ranges, 1.0)
        total_rooms = numeric_field("Total rooms", "total_rooms", ranges, 10.0)
        bedrooms_unknown = st.checkbox("Total bedrooms unknown")
        total_bedrooms = numeric_field("Total bedrooms", "total_bedrooms", ranges, 10.0, disabled=bedrooms_unknown)
    return {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": None if bedrooms_unknown else total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }


def predict_price(model, values):
    frame = pd.DataFrame([values], columns=RAW_FEATURE_COLUMNS)
    frame["total_bedrooms"] = pd.to_numeric(frame["total_bedrooms"], errors="coerce")
    return float(model.predict(frame)[0])


def render_prediction_tab(model, metadata):
    values = collect_inputs(metadata)
    if not st.button("Predict house price", type="primary"):
        return
    problems = validate_inputs(values)
    if problems:
        for problem in problems:
            st.error(problem)
        return
    try:
        price = predict_price(model, values)
    except Exception as error:
        st.error(f"The prediction could not be computed: {error}")
        return
    typical_error = metadata["test_metrics"]["MAE"]
    first, second, third = st.columns(3)
    first.metric("Estimated median house value", f"${price:,.0f}")
    second.metric("Typical error (test MAE)", f"±${typical_error:,.0f}")
    third.metric("Likely range", f"${max(price - typical_error, 0):,.0f} to ${price + typical_error:,.0f}")
    st.map(pd.DataFrame({"latitude": [values["latitude"]], "longitude": [values["longitude"]]}), zoom=6)


def render_performance_tab(metadata):
    comparison = pd.DataFrame(metadata["model_comparison"])
    st.write(f"Selected model: **{metadata['best_model']}**")
    metrics = metadata["test_metrics"]
    first, second, third = st.columns(3)
    first.metric("Test R²", f"{metrics['R2']:.3f}")
    second.metric("Test MAE", f"${metrics['MAE']:,.0f}")
    third.metric("Test RMSE", f"${metrics['RMSE']:,.0f}")
    figure = px.bar(comparison, x="model", y="R2", color="model", title="Test R² by model")
    figure.update_layout(showlegend=False)
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(comparison.set_index("model").round(4), use_container_width=True)


def render_importance_tab(metadata):
    importance = pd.Series(metadata["feature_importance"]).sort_values().reset_index()
    importance.columns = ["feature", "importance"]
    figure = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Permutation importance (decrease in R² when shuffled)",
    )
    st.plotly_chart(figure, use_container_width=True)


def main():
    st.title("🏠 House Price Predictor")
    st.caption("Predicts the median house value of a California housing district from its characteristics.")
    if not (MODEL_PATH.exists() and METADATA_PATH.exists()):
        st.error("model.pkl or metadata.json was not found. Run notebook.ipynb first to train and save the model.")
        st.stop()
    model = load_model()
    metadata = load_metadata()
    prediction_tab, performance_tab, importance_tab = st.tabs(["Prediction", "Model performance", "Feature importance"])
    with prediction_tab:
        render_prediction_tab(model, metadata)
    with performance_tab:
        render_performance_tab(metadata)
    with importance_tab:
        render_importance_tab(metadata)


main()
