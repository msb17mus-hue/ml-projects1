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

INTERNET_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def category_select(label, column, categories):
    return st.selectbox(label, categories[column], key=f"select_{column}")


def collect_inputs(metadata):
    categories = metadata["categories"]
    ranges = metadata["numeric_ranges"]
    values = {}
    customer_column, service_column, account_column = st.columns(3)
    with customer_column:
        st.subheader("Customer")
        values["gender"] = category_select("Gender", "gender", categories)
        values["SeniorCitizen"] = 1 if st.selectbox("Senior citizen", ["No", "Yes"]) == "Yes" else 0
        values["Partner"] = category_select("Has a partner", "Partner", categories)
        values["Dependents"] = category_select("Has dependents", "Dependents", categories)
        values["tenure"] = st.slider(
            "Tenure (months)", int(ranges["tenure"]["min"]), int(ranges["tenure"]["max"]), int(ranges["tenure"]["median"])
        )
    with service_column:
        st.subheader("Services")
        values["PhoneService"] = category_select("Phone service", "PhoneService", categories)
        values["MultipleLines"] = category_select("Multiple lines", "MultipleLines", categories)
        values["InternetService"] = category_select("Internet service", "InternetService", categories)
        for column in INTERNET_SERVICES:
            values[column] = category_select(column, column, categories)
    with account_column:
        st.subheader("Account")
        values["Contract"] = category_select("Contract", "Contract", categories)
        values["PaperlessBilling"] = category_select("Paperless billing", "PaperlessBilling", categories)
        values["PaymentMethod"] = category_select("Payment method", "PaymentMethod", categories)
        values["MonthlyCharges"] = st.number_input(
            "Monthly charges",
            min_value=float(ranges["MonthlyCharges"]["min"]),
            max_value=float(ranges["MonthlyCharges"]["max"]),
            value=float(ranges["MonthlyCharges"]["median"]),
            step=0.5,
        )
        manual_total = st.checkbox("Enter total charges manually")
        if manual_total:
            values["TotalCharges"] = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=float(ranges["TotalCharges"]["max"]) * 2,
                value=float(ranges["TotalCharges"]["median"]),
                step=10.0,
            )
        else:
            values["TotalCharges"] = round(values["tenure"] * values["MonthlyCharges"], 2)
            st.caption(f"Total charges estimated as tenure x monthly charges: {values['TotalCharges']:,.2f}")
    return values


def validate_inputs(values):
    problems = []
    if values["PhoneService"] == "No" and values["MultipleLines"] not in ("No phone service", "No"):
        problems.append("Multiple lines requires phone service.")
    if values["InternetService"] == "No":
        inconsistent = [column for column in INTERNET_SERVICES if values[column] == "Yes"]
        if inconsistent:
            problems.append("These services need internet service: " + ", ".join(inconsistent) + ".")
    if values["tenure"] > 0 and values["TotalCharges"] < values["MonthlyCharges"] * 0.5:
        problems.append("Total charges look too low for this tenure and monthly charge.")
    return problems


def predict_probability(model, values):
    frame = pd.DataFrame([values], columns=RAW_FEATURE_COLUMNS)
    return float(model.predict_proba(frame)[0, 1])


def risk_band(probability, threshold):
    if probability >= threshold:
        return "High", "#d62828"
    if probability >= threshold / 2:
        return "Medium", "#e9c46a"
    return "Low", "#2a9d8f"


def retention_actions(values):
    actions = []
    if values["Contract"] == "Month-to-month":
        actions.append("Offer an incentive to move to a one-year or two-year contract.")
    if values["tenure"] <= 12:
        actions.append("Include the customer in an onboarding or early-life care programme.")
    if values["InternetService"] != "No" and values["TechSupport"] == "No":
        actions.append("Offer a tech support add-on or a free trial of it.")
    if values["PaymentMethod"] == "Electronic check":
        actions.append("Encourage automatic payment with a small discount.")
    if not actions:
        actions.append("Maintain regular engagement and monitor satisfaction.")
    return actions


def render_gauge(probability, color):
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Churn probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
        )
    )
    figure.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
    st.plotly_chart(figure, use_container_width=True)


def render_prediction_tab(model, metadata):
    values = collect_inputs(metadata)
    if not st.button("Predict churn risk", type="primary"):
        return
    problems = validate_inputs(values)
    if problems:
        for problem in problems:
            st.error(problem)
        return
    try:
        probability = predict_probability(model, values)
    except Exception as error:
        st.error(f"The prediction could not be computed: {error}")
        return
    threshold = metadata["decision_threshold"]
    band, color = risk_band(probability, threshold)
    first, second = st.columns(2)
    with first:
        render_gauge(probability, color)
    with second:
        st.metric("Risk level", band)
        st.metric("Model decision", "Likely to churn" if probability >= threshold else "Likely to stay")
        st.caption(f"Decision threshold: {threshold:.2f}. Average churn rate in the training data: {metadata['overall_churn_rate']:.1%}.")
        st.write("**Suggested retention actions (general practice, not model output)**")
        for action in retention_actions(values):
            st.write(f"- {action}")


def render_performance_tab(metadata):
    st.write(f"Selected model: **{metadata['best_model']}**")
    metrics = metadata["test_metrics"]
    columns = st.columns(5)
    for column, name in zip(columns, ["recall", "precision", "f1", "roc_auc", "average_precision"]):
        column.metric(name.replace("_", " ").upper(), f"{metrics[name]:.3f}")
    comparison = pd.DataFrame(metadata["model_comparison"])
    figure = px.bar(comparison, x="model", y="cv_roc_auc", title="Cross-validated ROC AUC by model")
    st.plotly_chart(figure, use_container_width=True)
    curve = metadata["roc_curve"]
    roc_figure = go.Figure()
    roc_figure.add_trace(go.Scatter(x=curve["false_positive_rate"], y=curve["true_positive_rate"], name="Selected model"))
    roc_figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash")))
    roc_figure.update_layout(title="ROC curve (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(roc_figure, use_container_width=True)
    st.dataframe(comparison.set_index("model").round(4), use_container_width=True)


def render_drivers_tab(metadata):
    importance = pd.Series(metadata["feature_importance"]).head(12).sort_values().reset_index()
    importance.columns = ["feature", "importance"]
    st.plotly_chart(
        px.bar(importance, x="importance", y="feature", orientation="h", title="Most important input features (permutation importance)"),
        use_container_width=True,
    )
    drivers = pd.Series(metadata["churn_drivers"]).sort_values().reset_index()
    drivers.columns = ["feature", "coefficient"]
    drivers["direction"] = drivers["coefficient"].apply(lambda value: "Increases churn" if value > 0 else "Decreases churn")
    st.plotly_chart(
        px.bar(
            drivers,
            x="coefficient",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Increases churn": "#d62828", "Decreases churn": "#2a9d8f"},
            title="Logistic regression coefficients",
        ),
        use_container_width=True,
    )


def main():
    st.title("📉 Customer Churn Predictor")
    st.caption("Estimates how likely a telecom customer is to cancel the service and shows what drives churn.")
    if not (MODEL_PATH.exists() and METADATA_PATH.exists()):
        st.error("model.pkl or metadata.json was not found. Run notebook.ipynb first to train and save the model.")
        st.stop()
    model = load_model()
    metadata = load_metadata()
    prediction_tab, performance_tab, drivers_tab = st.tabs(["Prediction", "Model performance", "Churn drivers"])
    with prediction_tab:
        render_prediction_tab(model, metadata)
    with performance_tab:
        render_performance_tab(metadata)
    with drivers_tab:
        render_drivers_tab(metadata)


main()
