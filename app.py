import streamlit as st
import numpy as np
import joblib
import os

# Use relative paths to load the scaler and ensemble model from the 'model' folder
scaler_path = os.path.join('model', 'scaler.pkl')
ensemble_model_path = os.path.join('model', 'ensemble_model.pkl')

# Load the scaler and ensemble model
scaler = joblib.load(scaler_path)
ensemble_model = joblib.load(ensemble_model_path)

# Set custom page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Apply custom styling using Markdown for background color and layout
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main section of the app
st.title("🩺 Diabetes Prediction App")

st.write(
    """
    This app helps predict whether a person is likely to develop diabetes based on the input parameters. 
    Please fill in the information below and click **Predict**.
    """
)

# Organizing input fields in two columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=3)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0)

with col2:
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=120, value=33)

# Prepare the input array for prediction
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

# Scale the input data using the saved scaler
input_data_scaled = scaler.transform(input_data)

# Button to trigger prediction
if st.button('Predict'):
    # Make the prediction
    prediction = ensemble_model.predict(input_data_scaled)
    
    # Display the result with styling
    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is unlikely to have diabetes.")
