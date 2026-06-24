# Gurgaon Real Estate Price Prediction

Streamlit app for predicting Gurgaon real estate prices using `df.pkl` and `pipeline.pkl`.

## Setup

1. Copy `df.pkl` and `pipeline.pkl` into the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes

- `df.pkl` should be a pickled pandas DataFrame.
- `pipeline.pkl` should be a pickled scikit-learn pipeline or model.
- The app tries to infer model input features from `pipeline.feature_names_in_` or `df.pkl` columns.
