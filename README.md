# 🩺 Chronic Kidney Disease (CKD) Detection Using Machine Learning

This project is a **CKD prediction system** that uses machine learning to identify the likelihood of **Chronic Kidney Disease** in patients based on medical attributes. It provides early warnings, hospital recommendations, dietary suggestions, and rich data visualizations — all through a user-friendly **Streamlit web app**.

---

## 🌟 Key Features

- 🔍 **CKD Prediction** using a trained machine learning model (Random Forest)
- 📊 **Visualize uploaded patient data** with graphs and heatmaps
- 🏥 **City-based hospital suggestions** with contact, maps, and websites
- 🥗 **Personalized dietary recommendations** for kidney health
- 🔐 **Login system** for secure access

---

## 🧠 How It Works

You input various patient attributes (like age, blood pressure, sugar, creatinine, etc.)  
👉 The app processes your inputs and predicts whether the person is likely to have CKD  
👉 If CKD is detected, it recommends:
- Diet changes
- Nearby nephrology hospitals
- Health advice for management

---

## 🧪 Technologies Used

- **Python**
- **Scikit-learn** – ML model (Random Forest)
- **Streamlit** – for building the interactive web app
- **Pandas, NumPy** – for data handling
- **Matplotlib, Seaborn, Plotly** – for visualizations
- **Folium, Google Maps** – for hospital locations
- **Joblib** – to load trained models
- **FuzzyWuzzy** – for flexible city name matching

---

## 🚀 How to Publish & Run This Project

Follow these steps to run this project locally or deploy it online.

### ✅ 1. Fork or Clone This Repository

```bash
git clone https://github.com/your-username/ckd-detection.git
cd ckd-detection
## 🖥️ App Navigation Guide

Here’s how to explore the app once it’s running:

### 🔐 Login Page
- Enter **Username**: `admin`
- Enter **Password**: `Sulthan`
- Press **Login** to access the app

### 🧪 Predict CKD
- Input patient details (age, blood pressure, albumin, etc.)
- Click **Predict**
- Get instant feedback:
  - ✅ Healthy or ❌ CKD Detected
  - 📋 Treatment suggestions if CKD is detected

### 📍 Nearby Hospitals
- Enter your **city name**
- Get **hospital recommendations** (contact, website, location)
- Built-in support for cities like Chennai, Bangalore, Vellore, etc.
- Direct Google Maps link included

### 🥦 Dietary Recommendations
- Shows safe & restricted foods
- Tips on sodium, protein, potassium, and phosphorus
- Lifestyle advice for CKD management

### 📊 Data Visualization
- Upload your own **CKD dataset** (CSV format)
- See:
  - Correlation heatmaps
  - Bar graphs
  - KDE plots
  - Scatter plots
- Explore relationships between features & CKD status



