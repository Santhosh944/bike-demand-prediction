import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('model/lstm_model.h5', compile=False)
scaler = joblib.load('model/scaler.pkl')

st.title("🚲 Bike Rental Demand Prediction")

st.sidebar.header("Enter the Current Conditions")

hour = st.sidebar.slider("Hour of Day", 0, 23)
day = st.sidebar.selectbox("Day of Week", list(range(7)), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
temp = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
windspeed = st.sidebar.slider("Windspeed", 0.0, 60.0, 10.0)

# Prepare dummy sequence
input_data = np.array([[hour, day, temp, humidity, windspeed, 0]])
scaled = scaler.transform(input_data)

dummy_seq = np.zeros((1, 24, 5))
dummy_seq[0, -1, :] = scaled[0, :-1]  # insert features only (excluding target)

if st.button("Predict"):
    prediction = model.predict(dummy_seq)
    predicted_count = prediction[0][0] * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
    st.success(f"🔮 Predicted Bike Demand: **{int(predicted_count)} bikes**")
