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

```
Global-Air-Quality-Prediction/
│
├── .devcontainer/                 # VSCode Dev Container (optional)
├── LICENSE                        # MIT license
├── README.md                      # Project documentation
├── aqi_model.pkl                  # Trained XGBoost AQI prediction model
├── encoders.pkl                   # Label encoders for country & city
├── aqi_notebook.ipynb             # Jupyter notebook used for training
├── global_air_pollution_data.csv  # Historical dataset used for training
├── requirements.txt               # Python dependencies
└── streamlit_app.py               # Main Streamlit application
```
---

## 📊 Dataset

This project uses a global historical dataset containing pollutant concentrations and AQI values for multiple cities and countries.

The dataset features include:

- CO concentration  
- O₃ concentration  
- NO₂ concentration  
- PM2.5 concentration  
- AQI  
- Country  
- City


---

## 🧠 Model Details

- **Model Type:** `XGBoost Regressor`
- **Model Size:** `<1 MB` (aqi_encoder.pkl (~263 KB) + encoders.pkl (~200 KB))
- **Training Code:** `aqi_notebook.ipynb`

### **Input Features**
| Feature | Type |
|--------|------|
| CO (ppb) | Numeric |
| O₃ (ppb) | Numeric |
| NO₂ (ppb) | Numeric |
| PM2.5 (µg/m³) | Numeric |
| Country | Label-encoded |
| City | Label-encoded |

### **Performance**
- **MAE:** ~3 AQI points  
- **Very stable** on validation  
- **Fast inference** in Streamlit  

---


## ✨ Features

- 🌐 Predict AQI for any supported city and country  
- 📈 Input pollutant concentrations (with proper units)  
- ⚡ Fast ML predictions using XGBoost  
- 🎨 Clean & interactive Streamlit interface  
- 🧩 Includes saved label encoders for consistent predictions  
- 🔢 Converts output to AQI category  

---

## 📦 Requirements

- xgboost==2.1.1
- streamlit==1.32.0
- scikit-learn==1.7.2
- pandas==1.5.3
- numpy==1.23.5
- seaborn==0.12.2
- matplotlib==3.7.5
- pickle-mixin==1.0.2
- pillow==10.4.0

---

## 🖥️ Installation

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/your-username/Global-Air-Quality-Prediction.git
cd Global-Air-Quality-Prediction
```
### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```
### **3️⃣ Run the application**
```bash
streamlit run streamlit_app.py
```
