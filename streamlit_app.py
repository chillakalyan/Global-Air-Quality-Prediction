import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("global_air_pollution_data.csv")  
# Remove BOM, tabs, spaces, quotes from all column names
df.columns = (
    df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.strip()
)

# --------------------------
# Load model + encoders
# --------------------------
model = pickle.load(open("aqi_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

le_country = encoders["le_country"]
le_city = encoders["le_city"]

# --------------------------
# Title
# --------------------------
st.title("🌍 Global Air Quality Index (AQI) Prediction")
st.write("Predict the Air Quality Index using pollutant AQI values + location.")

# --------------------------
# Country dropdown (no cleaning required)
# --------------------------
country = st.selectbox(
    "Select Country",
    sorted(df["country_name"].unique())
)

# --------------------------
# City dropdown filtered by country
# --------------------------
filtered_cities = df[df["country_name"] == country]["city_name"].unique()
city = st.selectbox("Select City", sorted(filtered_cities))

# --------------------------
# Pollutant Inputs
# --------------------------
co = st.number_input("CO AQI Value", min_value=0.0)
ozone = st.number_input("Ozone AQI Value", min_value=0.0)
no2 = st.number_input("NO₂ AQI Value", min_value=0.0)
pm25 = st.number_input("PM2.5 AQI Value", min_value=0.0)

# --------------------------
# Encode
# --------------------------
country_encoded = le_country.transform([country])[0]
city_encoded = le_city.transform([city])[0]

# --------------------------
# Predict
# --------------------------
if st.button("Predict AQI"):
    features = np.array([[co, ozone, no2, pm25, country_encoded, city_encoded]])
    prediction = round(model.predict(features)[0], 2)

    st.success(f"Predicted AQI: {prediction}")

    # AQI category
    def classify(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"

    st.info(f"AQI Category: **{classify(prediction)}**")


