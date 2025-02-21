import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import pyttsx3  # Text-to-Speech (TTS)
import speech_recognition as sr  # Voice Input
import threading  # For running pyttsx3 asynchronously

# Load trained model, scaler, and encoders
model = joblib.load("crop_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Get unique values for dropdowns
soil_types = label_encoders["Soil_Type"].classes_
seasons = label_encoders["Season"].classes_
irrigation_types = label_encoders["Irrigation_Type"].classes_

# Function to speak text asynchronously
def speak_text(text):
    """Speak text asynchronously to avoid run loop errors in Streamlit."""
    def run():
        local_engine = pyttsx3.init()
        local_engine.say(text)
        local_engine.runAndWait()
    
    threading.Thread(target=run, daemon=True).start()

# Function for voice input
def get_voice_input(prompt_text):
    """Speak the prompt and get voice input."""
    speak_text(prompt_text)  # Speak prompt without blocking Streamlit

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(prompt_text)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio).strip().capitalize()
            st.success(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Sorry, I couldn't understand. Please try again.")
            return None
        except sr.RequestError:
            st.error("❌ Speech service unavailable. Check your internet connection.")
            return None

# Streamlit UI
st.title("🌾 AI Crop Recommendation Chatbot 🤖")

st.markdown(
    """
    👋 Hello! I'm here to recommend the best crop for you.  
    Enter the **soil type, season, irrigation type, and water usage**, and I'll suggest the best crop for your land. 🚜  
    You can either **select manually** or **use voice input** (click 🎤 buttons).  
    """
)

# User input options
col1, col2 = st.columns(2)

# Soil Type Input
with col1:
    soil_type = st.selectbox("🌍 Select Soil Type", soil_types)
with col2:
    if st.button("🎤 Speak Soil Type"):
        voice_input = get_voice_input("Please say the soil type.")
        if voice_input and voice_input in soil_types:
            soil_type = voice_input
        else:
            st.error("❌ Invalid input. Please select from the options.")

# Season Input
with col1:
    season = st.selectbox("⏳ Select Season", seasons)
with col2:
    if st.button("🎤 Speak Season"):
        voice_input = get_voice_input("Please say the season.")
        if voice_input and voice_input in seasons:
            season = voice_input
        else:
            st.error("❌ Invalid input. Please select from the options.")

# Irrigation Type Input
with col1:
    irrigation = st.selectbox("💧 Select Irrigation Type", irrigation_types)
with col2:
    if st.button("🎤 Speak Irrigation Type"):
        voice_input = get_voice_input("Please say the irrigation type.")
        if voice_input and voice_input in irrigation_types:
            irrigation = voice_input
        else:
            st.error("❌ Invalid input. Please select from the options.")

# Water Usage Input
water_usage = st.number_input("🚰 Enter Water Usage (cubic meters)", min_value=0.0)

if st.button("🎤 Speak Water Usage"):
    voice_input = get_voice_input("Please say the water usage in cubic meters.")
    try:
        water_usage = float(voice_input)
        st.success(f"✅ Water usage set to {water_usage} cubic meters")
    except ValueError:
        st.error("❌ Invalid number. Please try again.")

# Prediction button
if st.button("🧑‍🌾 Predict Crop"):
    with st.spinner("🤖 Thinking..."):
        time.sleep(2)  # Simulating processing time

    # Encode categorical inputs
    soil_encoded = label_encoders["Soil_Type"].transform([soil_type])[0]
    season_encoded = label_encoders["Season"].transform([season])[0]
    irrigation_encoded = label_encoders["Irrigation_Type"].transform([irrigation])[0]

    # Scale numerical input
    water_scaled = scaler.transform(np.array([[water_usage]]))[0][0]

    # Make prediction
    prediction = model.predict([[soil_encoded, season_encoded, water_scaled, irrigation_encoded]])
    predicted_crop = label_encoders["Crop_Type"].inverse_transform(prediction)[0]

    # Typing Effect
    message = f"🌱 Based on your inputs, the recommended crop is *{predicted_crop}*."
    display_text = ""
    for char in message:
        display_text += char
        st.markdown(display_text)
        time.sleep(0.05)  # Simulates typing effect

    # Convert text to speech
    speak_text(f"The recommended crop is {predicted_crop}.")

    st.success(f"🌱 Recommended Crop: *{predicted_crop}*")
