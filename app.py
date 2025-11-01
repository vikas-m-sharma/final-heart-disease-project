import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open('heart_disease_model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

# Define columns (same as training)
categorical_cols = ['Gender', 'ChestPainType', 'FastingBS', 'RestingECG',
                    'ExerciseAngina', 'ST_Slope', 'MajorVessels', 'Thalassemia']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'ST_Depression']

st.set_page_config(page_title="Heart Disease Prediction",
                   page_icon="🫀", layout="centered")

st.title("🩺 Heart Disease Prediction App")
st.write("This app predicts the likelihood of heart disease based on patient data.")

st.markdown("---")

# Collect inputs
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", 20, 100, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    RestingECG = st.selectbox("Resting ECG Results", [0, 1, 2])
    MaxHR = st.number_input("Maximum Heart Rate", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise-Induced Angina", [0, 1])
    ST_Depression = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    ST_Slope = st.selectbox("ST Slope", [0, 1, 2])
    MajorVessels = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia (1–3)", [1, 2, 3])

# Convert gender to numeric
Gender = 1 if Gender == "Male" else 0

# Create DataFrame
input_dict = {
    'Age': Age,
    'Gender': Gender,
    'ChestPainType': ChestPainType,
    'RestingBP': RestingBP,
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'RestingECG': RestingECG,
    'MaxHR': MaxHR,
    'ExerciseAngina': ExerciseAngina,
    'ST_Depression': ST_Depression,
    'ST_Slope': ST_Slope,
    'MajorVessels': MajorVessels,
    'Thalassemia': Thalassemia
}

input_df = pd.DataFrame([input_dict])

# One-hot encode the same columns as training
input_encoded = pd.get_dummies(
    input_df, columns=categorical_cols, drop_first=True)

# Get expected column order from model
expected_features = model.feature_names_in_

# Align columns (fill missing with 0)
input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

# Scale numerical features
input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

# Prediction
if st.button("🔍 Predict Heart Disease Risk"):
    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease detected!")
    else:
        st.success("✅ No signs of Heart Disease detected.")

st.caption("Developed by Vikas Sharma | © 2025 | Machine Learning Project")
