# 📘 Global Air Quality Prediction – AQI Predictor

[🚀 **Live Demo**](https://global-air-quality-prediction-nwkngwlkm7aq4apouo7mcp.streamlit.app/)

A machine-learning powered **Streamlit web application** that predicts the **Air Quality Index (AQI)** based on pollutant concentrations and geographic location (Country + City).

This project uses an **XGBoost regression model** trained on a historical global air-pollution dataset.  
The app supports real-time user inputs for major pollutants (PM2.5, CO, NO₂, O₃), encodes location features, and predicts both:

- **Numerical AQI Value**
- **AQI Category** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)

---

## 🌍 About the Project

Air quality plays a vital role in public health and climate assessment.  
This application predicts AQI using the following pollutant features:

### **Pollutants Used**
- PM2.5 (µg/m³)
- Carbon Monoxide – CO (ppb)
- Nitrogen Dioxide – NO₂ (ppb)
- Ozone – O₃ (ppb)
- Country
- City

### **Model Outputs**
- Predicted AQI value  
- AQI Category (Good → Severe)

---

## 📁 Repository Structure

Global-Air-Quality-Prediction/

│
├── .devcontainer/              # VSCode Devcontainer settings (optional)
├── LICENSE                     # MIT license
├── README.md                   # Project documentation
├── aqi_model.pkl               # Trained XGBoost AQI prediction model
├── encoders.pkl                # Label encoders for country & city
├── aqi_notebook.ipynb          # Jupyter notebook used for training
├── global_air_pollution_data.csv  # Historical dataset used for training
├── requirements.txt            # Python dependencies
└── streamlit_app.py            # Main Streamlit application

