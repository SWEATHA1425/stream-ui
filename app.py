import streamlit as st
import sounddevice as sd
import numpy as np
import pickle
import time
from scipy.io.wavfile import write
import tempfile
from utils import extract_features

# Load model
model = pickle.load(open("model.pkl", "rb"))

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
if st.button("🎙️ Record 10-Second Audio"):
    st.info("Recording started... Speak or scream now!")

    # Countdown (one line update)
    countdown_placeholder = st.empty()
    for i in range(duration, 0, -1):
        countdown_placeholder.markdown(f"⏳ **{i} seconds left...**")
        time.sleep(1)
    countdown_placeholder.empty()


    # Recording with progress bar
    progress_bar = st.progress(0)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    for i in range(100):
        time.sleep(duration / 100)
        progress_bar.progress(i + 1)
    sd.wait()
    st.success("✅ Recording complete!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write(tmpfile.name, fs, audio)
        st.audio(tmpfile.name)

        features = extract_features(tmpfile.name)
        probs = model.predict(features)[0]
        pred_index = np.argmax(probs)
        prediction = class_names[pred_index]

        st.subheader("Prediction Result:")
        if prediction == "Happy_Scream":
            st.success("😊 Happy Scream Detected!")
        elif prediction == "Screaming":
            st.markdown("<h1 style='color:red;'>🚨 DANGER DETECTED! 🚨</h1>", unsafe_allow_html=True)
            st.error("🚨 DANGER! Scream Detected!")
        else:
            st.info("🗣️ Normal Conversation")

        st.subheader("📊 Confidence Scores:")
        for i, label in enumerate(class_names):
            st.write(f"- {label}: `{probs[i]*100:.2f}%`")
