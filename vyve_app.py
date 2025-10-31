
import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
from geopy.geocoders import Nominatim
import cv2
import joblib

st.header("🛰️ Vyve Lab – Land Suitability Analyzer")

# Load model and scaler
model = tf.keras.models.load_model('vyve_ndvi_model.h5')
scaler = joblib.load('vyve_scaler.pkl')


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


option = st.radio(
    "Choose how you want to analyze land health:",
    ["📤 Upload Image", "📍 Enter ZIP Code"]
)

# ----------- 1️⃣ IMAGE UPLOAD MODE -----------
if option == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload a land or field image (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read and display
        image = np.array(Image.open(uploaded_file).convert('RGB'))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Compute vegetation index (simple Green Index)
        red = image[:, :, 0].astype(float)
        green = image[:, :, 1].astype(float)
        veg_index = (green - red) / (green + red + 1e-5)
        veg_norm = (veg_index - np.nanmin(veg_index)) / (np.nanmax(veg_index) - np.nanmin(veg_index))
        mean_veg = np.nanmean(veg_norm)

        st.write(f"🌱 **Average Vegetation Index:** {mean_veg:.3f}")

        if mean_veg > 0.6:
            st.success("🟢 This land looks healthy and suitable for vegetation!")
        elif mean_veg > 0.4:
            st.warning("🟡 Moderate vegetation — could improve with better irrigation or nutrients.")
        else:
            st.error("🔴 Poor vegetation potential — likely dry or stressed area.")

        st.image(veg_norm, caption="Vegetation Density Map", use_column_width=True, clamp=True)

# ----------- 2️⃣ ZIP CODE MODE -----------
elif option == "📍 Enter ZIP Code":
    zipcode = st.text_input("Enter ZIP Code (e.g., 21044):")

    if st.button("Analyze ZIP Code Area") and zipcode:
        geolocator = Nominatim(user_agent="vyve_lab")
        location = geolocator.geocode(zipcode)
        if location:
            lat, lon = location.latitude, location.longitude
            st.write(f"📍 Coordinates: {lat:.4f}, {lon:.4f}")

            # Fetch NASA POWER NDVI proxy data
            url = (
                f"https://power.larc.nasa.gov/api/temporal/daily/point?"
                f"parameters=T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN"
                f"&community=ag"
                f"&latitude={lat}&longitude={lon}"
                f"&start=20240601&end=20240630"
                f"&format=JSON"
            )
            resp = requests.get(url)
            data = resp.json().get("properties", {}).get("parameter", {})

            if data:
                import pandas as pd
                df = pd.DataFrame(data)
                df.index = pd.to_datetime(df.index)
                avg_temp = df["T2M_MAX"].mean()
                avg_precip = df["PRECTOTCORR"].mean()

                # Estimate vegetation suitability using a simple rule
                if avg_precip > 3 and avg_temp < 35:
                    st.success("🟢 High vegetation potential – suitable region for growth!")
                elif avg_precip > 1:
                    st.warning("🟡 Moderate suitability – irrigation may be needed.")
                else:
                    st.error("🔴 Low suitability – likely arid or low rainfall region.")
            else:
                st.error("⚠️ Could not fetch NASA data. Try another ZIP code.")
        else:
            st.error("❌ Invalid ZIP code or not found.")
