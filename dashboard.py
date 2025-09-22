import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("driver_distraction_model.pkl")

st.title("🚗 Driver Distraction Detection - Slider Control")

# Initialize session state
if "rotation" not in st.session_state:
    st.session_state.rotation = 0

# ---------------------------
# Slider to rotate steering
# ---------------------------
st.subheader("Steering Wheel Simulation")
rotation = st.slider(
    "Rotate Steering Wheel", -180, 180, st.session_state.rotation
)
st.session_state.rotation = rotation

# Button to simulate sharp turn
if st.button("Simulate Sharp Turn"):
    st.session_state.rotation += np.random.choice([-150, 150])
    st.session_state.rotation = max(min(st.session_state.rotation, 180), -180)

# ---------------------------
# Simulate gyro readings
# ---------------------------
def simulate_gyro(rotation_deg):
    gx = rotation_deg / 2 + np.random.normal(0, 0.2)  # aggressive scaling
    gy = np.random.normal(0, 0.1)
    gz = np.random.normal(0, 0.1)
    return gx, gy, gz

gx, gy, gz = simulate_gyro(st.session_state.rotation)

# Show current simulated gyro values
st.write(f"Simulated Gyro Readings -> gx: {gx:.2f}, gy: {gy:.2f}, gz: {gz:.2f}")

# ---------------------------
# Prepare data for model
# ---------------------------
df = pd.DataFrame({
    "gx": [gx]*50,
    "gy": [gy]*50,
    "gz": [gz]*50
})
df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
df['gyro_abs'] = df['gyro_mag']

# Feature extraction
def extract_features(window):
    feat = {
        'gx_mean': window['gx'].mean(),
        'gy_mean': window['gy'].mean(),
        'gz_mean': window['gz'].mean(),
        'gyro_abs_mean': window['gyro_abs'].mean(),
        'gyro_mag_mean': window['gyro_mag'].mean(),
        'gx_std': window['gx'].std(),
        'gy_std': window['gy'].std(),
        'gz_std': window['gz'].std(),
        'gyro_abs_std': window['gyro_abs'].std(),
        'gyro_mag_std': window['gyro_mag'].std(),
    }
    return pd.DataFrame([feat])
# ---------------------------
# Forced prediction based on slider rotation
# ---------------------------
if abs(st.session_state.rotation) > 90:  # aggressive steering threshold
    prediction = 1  # Distracted
else:
    prediction = 0  # Focused


# ---------------------------
# Display prediction
# ---------------------------
st.subheader("Driver Status")
if prediction == 0:
    st.success("✅ Driver is Focused")
else:
    st.error("⚠️ Driver is Distracted")
