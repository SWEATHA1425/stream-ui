import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import time
import tempfile
from utils import extract_features

# Load model
model = load_model("model.h5")

# Class labels (order must match your training)
class_names = ["Happy_Scream", "Normal_Conversation", "Screaming"]

st.set_page_config(page_title="Scream Detection", layout="centered")
st.title("🎤 Human Scream Detection App")
st.markdown("Record **10 seconds** of audio or upload a `.wav` file and predict:")
st.markdown("- 😊 Happy Scream\n- 🗣️ Normal Conversation\n- 🚨 Dangerous Scream")

duration = 10  # seconds
fs = 22050     # Sample rate

# Upload option
uploaded_file = st.file_uploader("Upload a WAV file or Record", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file)
    features = extract_features(uploaded_file)
    probs = model.predict(features)[0]
    pred_index = np.argmax(probs)
    prediction = class_names[pred_index]

    st.subheader("Prediction Result:")
    if prediction == "Happy_Scream":
        st.success("😊 Happy Scream Detected!")
    elif prediction == "Screaming":
        st.error("🚨 DANGER! Scream Detected!")
    else:
        st.info("🗣️ Normal Conversation")

    st.subheader("📊 Confidence Scores:")
    for i, label in enumerate(class_names):
        st.write(f"- {label}: `{probs[i]*100:.2f}%`")

# Record button

