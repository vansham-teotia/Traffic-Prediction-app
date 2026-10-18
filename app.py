import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Traffic Volume Forecaster", page_icon="🚦", layout="wide")

# 2. Custom CSS Injection
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Style the prediction button */
    .stButton>button {
        background-color: #005A9C; 
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #003F6E;
        color: white;
    }
    /* Custom Metric Card */
    .metric-card {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #005A9C;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_model():
    return joblib.load("traffic_volume_random_forest_refined.pkl")

model = load_model()

# 4. Header Section
st.title("🚦 Advanced Traffic Volume Forecasting")
st.markdown("Predict hourly vehicle traffic using machine learning and environmental metrics. Adjust the parameters in the sidebar to generate a new forecast.")
st.divider()

# 5. Sidebar Layout
with st.sidebar:
    st.header("⚙️ Forecast Parameters")
    
    with st.expander("📅 Time & Date Features", expanded=True):
        junction = st.selectbox("Junction ID", [1, 2, 3, 4]) 
        hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
        day_of_week = st.selectbox(
            "Day of Week", 
            options=[0, 1, 2, 3, 4, 5, 6], 
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
        )
        month = st.slider("Month (1-12)", 1, 12, 10)
        is_weekend = 1 if day_of_week >= 5 else 0

    with st.expander("🌤️ Environmental Conditions", expanded=True):
        temp = st.number_input("Temperature (°C)", value=25.0, step=0.5)
        humidity = st.number_input("Relative Humidity (%)", value=60.0, step=1.0)
        precip = st.number_input("Precipitation (mm)", value=0.0, step=0.1)
        wind = st.number_input("Wind Speed (km/h)", value=10.0, step=0.5)

    st.write("") # Spacing
    predict_btn = st.button("Generate Forecast")

# 6. Main Dashboard Layout
col1, col2 = st.columns([2, 1], gap="large")

if predict_btn:
    # Format inputs for the model
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
    
    # Generate prediction
    prediction = int(np.round(model.predict(input_data)[0]))
    
    # Display Results
    with col1:
        st.subheader("Forecast Results")
        st.markdown(f'''
            <div class="metric-card">
                <h3 style="color: #555; margin-bottom: 0;">Estimated Vehicle Volume</h3>
                <h1 style="color: #005A9C; font-size: 5rem; margin: 10px 0;">{prediction}</h1>
                <p style="color: #888; font-size: 1.2rem;">Vehicles per hour</p>
            </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.subheader("Active Parameters")
        # Clean up the dataframe for the summary table
        display_df = input_data.T.rename(columns={0: "Value"})
        st.dataframe(display_df, use_container_width=True)

else:
    st.info("👈 Please adjust the parameters in the sidebar and click **Generate Forecast** to see the results.")