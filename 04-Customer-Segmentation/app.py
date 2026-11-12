import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from features import CLUSTER_FEATURES

BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIRECTORY / "model.pkl"
METADATA_PATH = BASE_DIRECTORY / "metadata.json"

FIELD_LABELS = {
    "Age": "Age (years)",
    "AnnualIncome": "Annual income (k$)",
    "SpendingScore": "Spending score (1-100)",
}
ACTIONS = {
    ("High", "High"): "Premium loyalty programme, early access to new collections and personal service.",
    ("High", "Low"): "Investigate why a high-income group spends little: premium offers, better targeting and surveys.",
    ("Low", "High"): "Value bundles, discounts and loyalty points to keep a high-spending, budget-conscious group engaged.",
    ("Low", "Low"): "Low-cost awareness campaigns and entry-level promotions. Avoid heavy marketing spend.",
}

st.set_page_config(page_title="Customer Segmentation", page_icon="🧩", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def collect_inputs(metadata):
    ranges = metadata["feature_ranges"]
    values = {}
    columns = st.columns(len(CLUSTER_FEATURES))
    for column_area, column in zip(columns, CLUSTER_FEATURES):
        limits = ranges[column]
        with column_area:
            values[column] = st.number_input(
                FIELD_LABELS[column],
                min_value=float(limits["min"]),
                max_value=float(limits["max"]),
                value=float(limits["median"]),
                step=1.0,
                key=f"value_{column}",
            )
    return values


def render_prediction_tab(model, metadata):
    values = collect_inputs(metadata)
    if not st.button("Find segment", type="primary"):
        return
    try:
        frame = pd.DataFrame([values], columns=CLUSTER_FEATURES)
        segment = str(int(model.predict(frame)[0]))
    except Exception as error:
        st.error(f"The segment could not be computed: {error}")
        return
    levels = metadata["segment_levels"][segment]
    st.subheader(metadata["segment_names"][segment])
    profile = next(row for row in metadata["segment_profiles"] if str(row["segment"]) == segment)
    first, second, third, fourth = st.columns(4)
    first.metric("Customers in segment", f"{int(profile['customers'])}")
    second.metric("Average age", f"{profile['Age']:.1f}")
    third.metric("Average income (k$)", f"{profile['AnnualIncome']:.1f}")
    fourth.metric("Average spending score", f"{profile['SpendingScore']:.1f}")
    st.write("**Suggested marketing approach (general practice, not model output)**")
    st.write(ACTIONS[(levels["income"], levels["spending"])])


def render_explorer_tab(metadata):
    points = pd.DataFrame(metadata["customer_points"])
    points["segment"] = points["segment"].astype(str)
    x_column = st.selectbox("Horizontal axis", CLUSTER_FEATURES, index=1)
    y_column = st.selectbox("Vertical axis", CLUSTER_FEATURES, index=2)
    figure = px.scatter(points, x=x_column, y=y_column, color="segment", title="Customers coloured by segment")
    st.plotly_chart(figure, use_container_width=True)
    profiles = pd.DataFrame(metadata["segment_profiles"]).set_index("segment")
    st.dataframe(profiles, use_container_width=True)


def render_quality_tab(metadata):
    first, second, third = st.columns(3)
    first.metric("Segments", metadata["n_segments"])
    second.metric("Silhouette", f"{metadata['silhouette']:.3f}")
    third.metric("Stability (mean ARI, subsamples)", f"{metadata['stability']['subsample_ari_mean']:.3f}")
    selection = pd.DataFrame(metadata["k_selection"])
    st.plotly_chart(px.line(selection, x="k", y="silhouette", markers=True, title="Silhouette score by number of clusters"), use_container_width=True)
    st.plotly_chart(px.line(selection, x="k", y="inertia", markers=True, title="Elbow: inertia by number of clusters"), use_container_width=True)
    st.dataframe(pd.DataFrame(metadata["algorithm_comparison"]).set_index("algorithm").round(4), use_container_width=True)


def main():
    st.title("🧩 Customer Segmentation")
    st.caption("Assigns a mall customer to a behavioural segment using K-Means clustering on age, income and spending score.")
    if not (MODEL_PATH.exists() and METADATA_PATH.exists()):
        st.error("model.pkl or metadata.json was not found. Run notebook.ipynb first to train and save the model.")
        st.stop()
    model = load_model()
    metadata = load_metadata()
    prediction_tab, explorer_tab, quality_tab = st.tabs(["Find segment", "Segment explorer", "Model quality"])
    with prediction_tab:
        render_prediction_tab(model, metadata)
    with explorer_tab:
        render_explorer_tab(metadata)
    with quality_tab:
        render_quality_tab(metadata)


main()
