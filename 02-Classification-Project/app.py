import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import RAW_FEATURE_COLUMNS

BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIRECTORY / "model.pkl"
METADATA_PATH = BASE_DIRECTORY / "metadata.json"

FIELD_LABELS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Plasma glucose (mg/dL)",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skin fold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/ml)",
    "BMI": "Body mass index (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age (years)",
}
FIELD_STEPS = {"Pregnancies": 1.0, "Age": 1.0, "DiabetesPedigreeFunction": 0.01, "BMI": 0.1}

st.set_page_config(page_title="Diabetes Risk Classifier", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def collect_inputs(metadata):
    ranges = metadata["numeric_ranges"]
    optional_columns = set(metadata["zero_as_missing_columns"])
    values = {}
    columns = st.columns(2)
    for index, column in enumerate(RAW_FEATURE_COLUMNS):
        limits = ranges[column]
        with columns[index % 2]:
            unknown = False
            if column in optional_columns:
                unknown = st.checkbox(f"{FIELD_LABELS[column]}: not available", key=f"unknown_{column}")
            value = st.number_input(
                FIELD_LABELS[column],
                min_value=float(limits["min"]),
                max_value=float(limits["max"]),
                value=float(limits["median"]),
                step=FIELD_STEPS.get(column, 1.0),
                disabled=unknown,
                key=f"value_{column}",
            )
            values[column] = None if unknown else value
    return values


def predict_probability(model, values):
    frame = pd.DataFrame([values], columns=RAW_FEATURE_COLUMNS).apply(pd.to_numeric, errors="coerce")
    return float(model.predict_proba(frame)[0, 1])


def risk_band(probability):
    if probability < 0.3:
        return "Low", "#2a9d8f"
    if probability < 0.6:
        return "Moderate", "#e9c46a"
    return "High", "#d62828"


def render_gauge(probability, color):
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Estimated probability of diabetes"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
        )
    )
    figure.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
    st.plotly_chart(figure, use_container_width=True)


def render_prediction_tab(model, metadata):
    values = collect_inputs(metadata)
    if not st.button("Predict", type="primary"):
        return
    known_measurements = [value for value in values.values() if value is not None]
    if len(known_measurements) < 4:
        st.error("Please provide at least four of the eight measurements.")
        return
    try:
        probability = predict_probability(model, values)
    except Exception as error:
        st.error(f"The prediction could not be computed: {error}")
        return
    label, color = risk_band(probability)
    threshold = metadata["decision_threshold"]
    first, second = st.columns([1, 1])
    with first:
        render_gauge(probability, color)
    with second:
        st.metric("Risk level", label)
        st.metric("Model decision", "Diabetes" if probability >= threshold else "No diabetes")
        st.caption("This tool is an educational demonstration and does not replace medical advice.")
        missing = [FIELD_LABELS[column] for column, value in values.items() if value is None]
        if missing:
            st.info("Estimated without: " + ", ".join(missing))


def render_performance_tab(metadata):
    st.write(f"Selected model: **{metadata['best_model']}**")
    metrics = metadata["test_metrics"]
    columns = st.columns(5)
    for column, name in zip(columns, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        column.metric(name.replace("_", " ").upper(), f"{metrics[name]:.3f}")
    comparison = pd.DataFrame(metadata["model_comparison"])
    figure = px.bar(
        comparison.melt(id_vars="model", value_vars=["accuracy", "recall", "f1", "roc_auc"], var_name="metric", value_name="score"),
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Test-set performance by model",
    )
    st.plotly_chart(figure, use_container_width=True)
    curve = metadata["roc_curve"]
    roc_figure = go.Figure()
    roc_figure.add_trace(go.Scatter(x=curve["false_positive_rate"], y=curve["true_positive_rate"], name="Selected model"))
    roc_figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash")))
    roc_figure.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(roc_figure, use_container_width=True)
    st.dataframe(comparison.set_index("model").round(4), use_container_width=True)


def render_importance_tab(metadata):
    importance = pd.Series(metadata["feature_importance"]).sort_values().reset_index()
    importance.columns = ["feature", "importance"]
    figure = px.bar(importance, x="importance", y="feature", orientation="h", title="Permutation importance (drop in ROC AUC)")
    st.plotly_chart(figure, use_container_width=True)


def main():
    st.title("🩺 Diabetes Risk Classifier")
    st.caption("Estimates the probability of diabetes from diagnostic measurements (Pima Indians Diabetes dataset).")
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
