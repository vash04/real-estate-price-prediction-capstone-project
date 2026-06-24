import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_FILE = Path("df.pkl")
MODEL_FILE = Path("pipeline.pkl")


@st.cache_data
def load_dataframe():
    if not DATA_FILE.exists():
        return None
    try:
        return pd.read_pickle(DATA_FILE)
    except Exception as exc:
        st.error(f"df.pkl load failed: {exc}")
        return None


@st.cache_resource
def load_pipeline():
    if not MODEL_FILE.exists():
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        st.error(f"pipeline.pkl load failed: {exc}")
        return None


def infer_feature_names(data: pd.DataFrame | None, model) -> list[str]:
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if data is None:
        return []

    features = list(data.columns)
    if "price" in features:
        features.remove("price")
    return features


def render_input_widget(feature: str, series: pd.Series) -> object:
    series = series.dropna()
    if len(series) == 0:
        return st.text_input(feature, value="")

    if pd.api.types.is_numeric_dtype(series):
        minimum = float(series.min())
        maximum = float(series.max())
        default = float(series.median())
        if series.dtype.kind in "iu":
            return st.number_input(feature, min_value=int(minimum), max_value=int(maximum), value=int(default), step=1)
        return st.number_input(feature, min_value=minimum, max_value=maximum, value=default, step=0.1)

    options = sorted(series.astype(str).unique())
    return st.selectbox(feature, options)


def collect_inputs(features: list[str], data: pd.DataFrame | None) -> dict[str, object]:
    inputs: dict[str, object] = {}
    for feature in features:
        if data is not None and feature in data.columns:
            inputs[feature] = render_input_widget(feature, data[feature])
        else:
            inputs[feature] = st.text_input(feature)
    return inputs


def main() -> None:
    st.set_page_config(page_title="Gurgaon Real Estate Price Prediction", page_icon="🏠", layout="centered")
    st.title("Gurgaon Real Estate Price Prediction")
    st.write(
        "Upload `df.pkl` and `pipeline.pkl` in this folder, then use the form below to predict property price for Gurgaon."
    )

    data = load_dataframe()
    model = load_pipeline()

    if data is None:
        st.warning("`df.pkl` not found or could not be loaded. Please add it to the project folder.")
    if model is None:
        st.warning("`pipeline.pkl` not found or could not be loaded. Please add it to the project folder.")

    if data is not None:
        with st.expander("Dataset preview"):
            st.dataframe(data.head(10))
            st.write("Columns:", list(data.columns))
            st.write("Shape:", data.shape)

    feature_names = infer_feature_names(data, model)
    if not feature_names:
        st.warning("Unable to infer model input features. Ensure the pipeline exposes `feature_names_in_` or df.pkl contains feature columns.")
        return

    st.subheader("Enter feature values")
    with st.form("prediction_form"):
        user_inputs = collect_inputs(feature_names, data)
        submit = st.form_submit_button("Predict Price")

    if submit:
        try:
            input_df = pd.DataFrame([user_inputs])
            prediction = model.predict(input_df)
            price = float(prediction[0])
            st.success(f"Estimated price: ₹{price:,.2f}")
            st.write("Model input:", input_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.write("Model object:", model)

    with st.expander("Model information"):
        if model is not None:
            st.write(type(model))
            st.write(str(model))
        else:
            st.write("No model loaded yet.")


if __name__ == "__main__":
    main()
