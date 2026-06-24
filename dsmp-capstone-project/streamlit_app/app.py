import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SEARCH_PATHS = [APP_DIR, REPO_ROOT, REPO_ROOT.parent, Path.cwd()]
DATA_FILE_NAME = "df.pkl"
MODEL_FILE_NAME = "pipeline.pkl"
PREDICTION_RANGE_DELTA = 0.22


def find_file(name: str) -> Path | None:
    for folder in SEARCH_PATHS:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


@st.cache_data
def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


@st.cache_resource
def load_pipeline(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


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
        "This app predicts Gurgaon real estate prices. Place `df.pkl` and `pipeline.pkl` in the repository root or the streamlit_app folder."
    )

    data_path = find_file(DATA_FILE_NAME)
    model_path = find_file(MODEL_FILE_NAME)

    if data_path is None:
        st.warning("`df.pkl` not found. Copy this file into the repository root or the streamlit_app folder.")
    if model_path is None:
        st.warning("`pipeline.pkl` not found. Copy this file into the repository root or the streamlit_app folder.")

    data = None
    model = None

    if data_path is not None:
        try:
            data = load_dataframe(data_path)
        except Exception as exc:
            st.error(f"Error loading df.pkl: {exc}")

    if model_path is not None:
        try:
            model = load_pipeline(model_path)
        except Exception as exc:
            st.error(f"Error loading pipeline.pkl: {exc}")

    if data is not None:
        with st.expander("Dataset preview"):
            st.dataframe(data.head(10))
            st.write("Columns:", list(data.columns))
            st.write("Shape:", data.shape)

    feature_names = infer_feature_names(data, model)
    if not feature_names:
        st.warning("Unable to infer input features. Make sure `pipeline.pkl` exposes `feature_names_in_` or that `df.pkl` contains feature columns.")
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
            lower = price - PREDICTION_RANGE_DELTA
            upper = price + PREDICTION_RANGE_DELTA
            st.success(f"Estimated price: ₹{price:,.2f}")
            st.info(f"Prediction range: ₹{lower:,.2f} to ₹{upper:,.2f}")
            st.write("Model input:", input_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.write("Input:", user_inputs)
            st.write("Model:", model)

    with st.expander("Model information"):
        if model is not None:
            st.write(type(model))
            st.write(str(model))
        else:
            st.write("No model loaded.")


if __name__ == "__main__":
    main()
