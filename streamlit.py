import streamlit as st
import pickle
import numpy as np

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

# User Inputs
country = st.selectbox("Select Country", le_country.classes_)
city = st.selectbox("Select City", le_city.classes_)

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
    features = np.array([[co, ozone, no2, pm25, country_encoded, city_encoded]])
    prediction = model.predict(features)[0]

    # Display result
    st.success(f"Predicted AQI: {round(prediction, 2)}")

    # Category Logic
    def classify(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"

    category = classify(prediction)
    st.info(f"AQI Category: **{category}**")
