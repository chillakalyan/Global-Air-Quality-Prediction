import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --------------------------
# Load Dataset (For Country → City filtering)
# --------------------------
df = pd.read_csv("global_air_pollution_data.csv")

# Clean column names for safety
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("\t", "", regex=False)
)
# Clean weird characters from column names (tabs, quotes, dots)
df.columns = (
    df.columns.str.replace('"', "", regex=False)
              .str.replace("\t", "", regex=False)
              .str.replace(" ", "_")
              .str.lower()
              .str.strip()
)

# Fix inconsistent pollutant column names
rename_map = {
    "co_aqi_value": "co_aqi_value",
    "ozone_aqi_value": "ozone_aqi_value",
    "no2_aqi_value": "no2_aqi_value",
    "pm2.5_aqi_value": "pm25_aqi_value"
}

df = df.rename(columns=rename_map)

# --------------------------
# Load Model + Encoders
# --------------------------
model = pickle.load(open("aqi_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

le_country = encoders["le_country"]
le_city = encoders["le_city"]

# --------------------------
# Streamlit App UI
# --------------------------
st.title("🌍 Global Air Quality Index (AQI) Prediction")
st.write("Predict the Air Quality Index using pollutant AQI values + location.")

# --------------------------
# Country Dropdown
# --------------------------
country = st.selectbox(
    "Select Country",
    sorted(df["country_name"].unique())
)

# --------------------------
# City Dropdown Filtered by Country
# --------------------------
filtered_cities = df[df["country_name"] == country]["city_name"].unique()

city = st.selectbox(
    "Select City",
    sorted(filtered_cities)
)

# --------------------------
# Pollutant Inputs
# --------------------------
co = st.number_input("CO AQI Value", min_value=0.0)
ozone = st.number_input("Ozone AQI Value", min_value=0.0)
no2 = st.number_input("NO₂ AQI Value", min_value=0.0)
pm25 = st.number_input("PM2.5 AQI Value", min_value=0.0)

# --------------------------
# Encode Inputs
# --------------------------
country_encoded = le_country.transform([country])[0]
city_encoded = le_city.transform([city])[0]

# --------------------------
# Predict
# --------------------------
if st.button("Predict AQI"):
    # Prepare feature array in the same order as training
    features = np.array([[co, ozone, no2, pm25, country_encoded, city_encoded]])
    
    prediction = model.predict(features)[0]
    prediction = round(prediction, 2)

    # Display AQI number
    st.success(f"Predicted AQI: {prediction}")

    # --------------------------
    # Category Logic
    # --------------------------
    def classify(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Satisfactory"
        elif aqi <= 200:
            return "Moderate"
        elif aqi <= 300:
            return "Poor"
        elif aqi <= 400:
            return "Very Poor"
        else:
            return "Severe"

    category = classify(prediction)
    st.info(f"AQI Category: **{category}**")

