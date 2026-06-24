# Gurgaon Real Estate Price Prediction App

Ye folder ek Streamlit web app contain karta hai jo Gurgaon real estate price prediction ke liye banaaya gaya hai.

## Usage

1. `df.pkl` aur `pipeline.pkl` ko repository root ya `streamlit_app` folder mein copy karo.
2. Install dependencies:
   ```bash
   pip install -r streamlit_app/requirements.txt
   ```
3. Run karo:
   ```bash
   streamlit run streamlit_app/app.py
   ```

## Notes

- `df.pkl` ek pickled pandas DataFrame hona chahiye.
- `pipeline.pkl` ek pickled scikit-learn pipeline/model hona chahiye.
- App model ke `feature_names_in_` attribute se input features infer karega.
