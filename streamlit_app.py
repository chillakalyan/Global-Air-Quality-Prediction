# streamlit_app.py (overwrite existing file)
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------- Config ----------
CSV_PATH = "global_air_pollution_data.csv"  # must be in repo root
MODEL_PATH = "aqi_model.pkl"
ENC_PATH = "encoders.pkl"

# ---------- Load dataset ----------
@st.cache_data(show_spinner=False)
def load_dataset(path):
    df = pd.read_csv(path, dtype=str)  # load as strings first for robust cleaning
    # Strip BOM, quotes and whitespace from column names
    df.columns = (
        df.columns.str.replace("\ufeff", "", regex=False)
                  .str.replace('"', "", regex=False)
                  .str.replace("\t", "", regex=False)
                  .str.strip()
    )
    # Standardize common column names (lower)
    df.columns = df.columns.str.lower()
    # Fix pm2.5 naming variants to a consistent name
    df = df.rename(columns={
        "pm2.5_aqi_value": "pm25_aqi_value",
        "pm2_5_aqi_value": "pm25_aqi_value"
    })
    # Clean country and city columns: drop empty or nan-like rows
    if "country_name" in df.columns:
        df["country_name"] = df["country_name"].astype(str).str.strip()
        df = df[df["country_name"].str.lower() != "nan"]
        df = df[df["country_name"].str.strip() != ""]
    if "city_name" in df.columns:
        df["city_name"] = df["city_name"].astype(str).str.strip()
        df = df[df["city_name"].str.lower() != "nan"]
        df = df[df["city_name"].str.strip() != ""]
    return df

df = load_dataset(CSV_PATH)

# ---------- Load model and encoders ----------
@st.cache_resource(show_spinner=False)
def load_model_and_encoders(mpath, encpath):
    model = pickle.load(open(mpath, "rb"))
    encoders = pickle.load(open(encpath, "rb"))
    return model, encoders

model, encoders = load_model_and_encoders(MODEL_PATH, ENC_PATH)
le_country = encoders["le_country"]
le_city = encoders["le_city"]

# ---------- UI ----------
st.set_page_config(page_title="Global AQI Predictor", layout="centered")
st.title("🌍 Global Air Quality Index (AQI) Prediction")
st.write("Predict the AQI using pollutant AQI values + location.")

# Detect correct column names robustly
country_col = next((c for c in df.columns if "country" in c), "country_name")
city_col = next((c for c in df.columns if "city" in c), "city_name")

# Country dropdown (safe)
countries = sorted(df[country_col].unique())
country = st.selectbox("Select Country", countries)

# Filter cities for the chosen country
filtered_cities = sorted(df[df[country_col] == country][city_col].unique())
city = st.selectbox("Select City", filtered_cities)

# Numeric inputs
co = st.number_input("CO AQI Value", min_value=0.0, value=0.0, step=1.0)
ozone = st.number_input("Ozone AQI Value", min_value=0.0, value=0.0, step=1.0)
no2 = st.number_input("NO₂ AQI Value", min_value=0.0, value=0.0, step=1.0)
pm25 = st.number_input("PM2.5 AQI Value", min_value=0.0, value=0.0, step=1.0)

# Encode and predict safely
try:
    country_encoded = le_country.transform([country])[0]
    city_encoded = le_city.transform([city])[0]
except Exception as e:
    st.error("Selected country/city not found in encoder classes. Please retrain with consistent encoders.")
    st.stop()

if st.button("Predict AQI"):
    features = np.array([[co, ozone, no2, pm25, country_encoded, city_encoded]])
    pred = model.predict(features)[0]
    pred = float(pred)
    st.success(f"Predicted AQI: {pred:.2f}")

    def classify(aqi):
        if aqi <= 50: return "Good"
        if aqi <= 100: return "Satisfactory"
        if aqi <= 200: return "Moderate"
        if aqi <= 300: return "Poor"
        if aqi <= 400: return "Very Poor"
        return "Severe"

    st.info(f"AQI Category: **{classify(pred)}**")
