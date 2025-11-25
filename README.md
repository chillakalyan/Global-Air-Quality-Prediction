# 📘 Global Air Quality Prediction – AQI Predictor

📘 Global Air Quality Prediction – AQI Predictor

A machine-learning powered Streamlit web application that predicts the Air Quality Index (AQI) based on pollutant concentration values and geographical location (Country + City).

This project uses an XGBoost regression model, trained on historical global air-pollution datasets, along with Label Encoders for categorical features.
The deployed Streamlit UI allows users to interactively enter pollutant levels and obtain the predicted AQI and its corresponding category.

🌍 About the Project

Air quality plays a critical role in public health, environmental monitoring, and policy-making.
This project predicts AQI using the following pollutants:

PM2.5 (µg/m³)

NO₂ (ppb)

Ozone O₃ (ppb)

Carbon Monoxide CO (ppb)

Country

City

The model outputs:

Predicted AQI value

AQI Category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)

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

