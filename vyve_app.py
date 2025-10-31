import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model('vyve_ndvi_model.h5')
scaler = joblib.load('vyve_scaler.pkl')

st.title("🌾 Vyve Lab – Smart Agriculture AI")
st.markdown("Predict crop health (NDVI) based on weather conditions.")

# User input fields
tmax = st.number_input("Max Temperature (°C)", value=30.0)
tmin = st.number_input("Min Temperature (°C)", value=20.0)
precip = st.number_input("Precipitation (mm)", value=5.0)
solar = st.number_input("Solar Radiation (MJ/m²)", value=25.0)

if st.button("Predict NDVI"):
    X = np.array([[tmax, tmin, precip, solar]])
    X_scaled = scaler.transform(X)
    ndvi_pred = model.predict(X_scaled)[0][0]

    st.success(f"Predicted NDVI: {ndvi_pred:.3f}")

    if ndvi_pred > 0.6:
        st.markdown("🟢 **Healthy vegetation!** 🌱")
    elif ndvi_pred > 0.4:
        st.markdown("🟡 **Moderate health** – monitor conditions.")
    else:
        st.markdown("🔴 **Stressed vegetation** ⚠️ – possible drought or nutrient issue.")
