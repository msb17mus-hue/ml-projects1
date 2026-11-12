import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import MAX_CHARACTERS, MIN_WORDS, find_signal_terms, tokenize

BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIRECTORY / "model.pkl"
METADATA_PATH = BASE_DIRECTORY / "metadata.json"
EXAMPLE_REVIEW = "The acting was wonderful and the story was moving, but the ending felt a little rushed."

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def validate_review(text):
    problems = []
    if not isinstance(text, str):
        return ["Please enter a review in English before analysing the sentiment."]
    if len(tokenize(text)) < MIN_WORDS:
        problems.append(f"Please write at least {MIN_WORDS} words so that the model has something to work with.")
    if len(text) > MAX_CHARACTERS:
        problems.append(f"The review is longer than {MAX_CHARACTERS:,} characters. Please shorten it.")
    return problems


def render_gauge(probability):
    color = "#2a9d8f" if probability >= 0.5 else "#d62828"
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Probability that the review is positive"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
        )
    )
    figure.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
    st.plotly_chart(figure, use_container_width=True)


def render_prediction_tab(model, metadata):
    text = st.text_area("Paste or write a movie review (English)", value=EXAMPLE_REVIEW, height=180)
    if not st.button("Analyse sentiment", type="primary"):
        return
    problems = validate_review(text)
    if problems:
        for problem in problems:
            st.error(problem)
        return
    try:
        probability = float(model.predict_proba([text])[0, 1])
    except Exception as error:
        st.error(f"The prediction could not be computed: {error}")
        return
    positive_terms, negative_terms = find_signal_terms(text, metadata["term_weights"])
    first, second = st.columns(2)
    with first:
        render_gauge(probability)
    with second:
        st.metric("Prediction", "Positive" if probability >= 0.5 else "Negative")
        confidence = max(probability, 1 - probability)
        st.metric("Model confidence", f"{confidence:.1%}")
        if confidence < 0.65:
            st.info("The model is unsure about this review. Mixed or sarcastic reviews are hard for this kind of model.")
        st.caption("Trained on English movie reviews only. Other kinds of text may be scored unreliably.")
    left, right = st.columns(2)
    with left:
        st.write("**Words and phrases pushing towards positive**")
        if positive_terms:
            for term, weight in positive_terms:
                st.write(f"- {term} ({weight:+.2f})")
        else:
            st.write("None of the strongest known terms were found.")
    with right:
        st.write("**Words and phrases pushing towards negative**")
        if negative_terms:
            for term, weight in negative_terms:
                st.write(f"- {term} ({weight:+.2f})")
        else:
            st.write("None of the strongest known terms were found.")
    st.caption("Term weights come from the logistic regression model trained in the notebook and are an explanation aid, not the exact computation of the selected model.")


def render_performance_tab(metadata):
    st.write(f"Selected model: **{metadata['best_model']}**")
    metrics = metadata["test_metrics"]
    columns = st.columns(5)
    for column, name in zip(columns, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        column.metric(name.replace("_", " ").upper(), f"{metrics[name]:.3f}")
    comparison = pd.DataFrame(metadata["model_comparison"])
    melted = comparison.melt(id_vars="model", value_vars=["accuracy", "f1", "roc_auc"], var_name="metric", value_name="score")
    st.plotly_chart(px.bar(melted, x="model", y="score", color="metric", barmode="group", title="Test-set performance by model"), use_container_width=True)
    matrix = metadata["confusion_matrix"]
    heatmap = px.imshow(
        matrix,
        x=["Predicted negative", "Predicted positive"],
        y=["Actual negative", "Actual positive"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion matrix (test set)",
    )
    st.plotly_chart(heatmap, use_container_width=True)
    curve = metadata["roc_curve"]
    roc_figure = go.Figure()
    roc_figure.add_trace(go.Scatter(x=curve["false_positive_rate"], y=curve["true_positive_rate"], name="Selected model"))
    roc_figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash")))
    roc_figure.update_layout(title="ROC curve (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(roc_figure, use_container_width=True)
    st.dataframe(comparison.set_index("model").round(4), use_container_width=True)


def render_vocabulary_tab(metadata):
    terms = metadata["display_terms"]
    rows = [{"term": term, "weight": weight, "direction": "Positive"} for term, weight in terms["positive"].items()]
    rows += [{"term": term, "weight": weight, "direction": "Negative"} for term, weight in terms["negative"].items()]
    table = pd.DataFrame(rows).sort_values("weight")
    figure = px.bar(
        table,
        x="weight",
        y="term",
        orientation="h",
        color="direction",
        color_discrete_map={"Positive": "#2a9d8f", "Negative": "#d62828"},
        title="Strongest terms in the logistic regression model",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(f"Vocabulary size: {metadata['vocabulary_size']:,} terms. Median review length: {metadata['median_review_words']:.0f} words.")


def main():
    st.title("🎬 Movie Review Sentiment")
    st.caption("Classifies a movie review as positive or negative with TF-IDF features and a linear classifier.")
    if not (MODEL_PATH.exists() and METADATA_PATH.exists()):
        st.error("model.pkl or metadata.json was not found. Run notebook.ipynb first to train and save the model.")
        st.stop()
    model = load_model()
    metadata = load_metadata()
    prediction_tab, performance_tab, vocabulary_tab = st.tabs(["Analyse a review", "Model performance", "Vocabulary"])
    with prediction_tab:
        render_prediction_tab(model, metadata)
    with performance_tab:
        render_performance_tab(metadata)
    with vocabulary_tab:
        render_vocabulary_tab(metadata)


main()
