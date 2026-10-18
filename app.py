import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Traffic Volume Forecaster", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    # This loads the model you already trained in your notebook
    return joblib.load("traffic_volume_random_forest_refined.pkl")

model = load_model()

# App UI headers
st.title("🚦 Hourly Traffic Volume Forecaster")
st.write("Predict future traffic volumes based on time and weather conditions.")

# Create input sections
st.sidebar.header("Input Parameters")

# Temporal Inputs
st.sidebar.subheader("Time & Date Features")
junction = st.sidebar.selectbox("Junction", [1, 2, 3, 4]) 
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Day of Week (0=Monday, 6=Sunday)", [0, 1, 2, 3, 4, 5, 6])
month = st.sidebar.slider("Month", 1, 12, 10)
is_weekend = 1 if day_of_week >= 5 else 0

# Weather Inputs
st.sidebar.subheader("Weather Conditions")
temp = st.sidebar.number_input("Temperature (°C)", value=25.0)
humidity = st.sidebar.number_input("Relative Humidity (%)", value=60.0)
precip = st.sidebar.number_input("Precipitation (mm)", value=0.0)
wind = st.sidebar.number_input("Wind Speed (km/h)", value=10.0)

# Prediction Button
if st.button("Predict Traffic Volume"):
    # Format inputs into a DataFrame
    input_data = pd.DataFrame({
        "Junction": [junction],
        "Hour": [hour],
        "DayOfWeek": [day_of_week],
        "Month": [month],
        "Is_Weekend": [is_weekend],
        "temperature_2m (°C)": [temp],
        "relative_humidity_2m (%)": [humidity],
        "precipitation (mm)": [precip],
        "wind_speed_10m (km/h)": [wind]
    })
    
    # Run prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success(f"Predicted Traffic Volume: **{int(np.round(prediction))} vehicles**")