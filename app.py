import streamlit as st
import numpy as np
import librosa
import joblib
import sounddevice as sd
import wavio
import os
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Voice Emotion Detector", 
    page_icon="🎤", 
    layout="wide"
)

# --- CSS Styling (Glassmorphism & Neon Glow) ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #1e293b, #0f172a); color: white; }
    .main-title { text-align: center; font-size: 50px; font-weight: 900; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .emotion-box { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(20px); border: 1px solid #38bdf8; padding: 25px; border-radius: 30px; text-align: center; font-size: 30px; color: #fff; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    div.stButton > button { background: transparent; color: white; border: 2px solid #38bdf8; padding: 12px 25px; border-radius: 50px; font-weight: 700; transition: 0.3s; width: 100%; }
    div.stButton > button:hover { background: #38bdf8; transform: scale(1.05); box-shadow: 0 0 20px rgba(56, 189, 248, 0.5); }
    </style>
""", unsafe_allow_html=True)

# --- Load Trained Models ---
# Note: Ensure emotion_model.pkl and label_encoder.pkl are in the same folder
model = joblib.load("emotion_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# --- Feature Extraction Function ---
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack((mfccs, chroma, mel))

# --- Prediction & Display Function ---
def process_and_predict(file_path):
    with st.spinner('AI analyzing your voice...'):
        features = np.expand_dims(extract_features(file_path), axis=0)
        probs = model.predict_proba(features)[0]
        emotion = encoder.inverse_transform([np.argmax(probs)])[0]
        
        emoji = {"happy": "😊", "sad": "😢", "angry": "😡", "fearful": "😨", "neutral": "😐"}.get(emotion, "🎭")
        
        st.markdown(f'<div class="emotion-box">{emoji} Detected Emotion: {emotion.upper()}</div>', unsafe_allow_html=True)
        
        # Fireworks effect for happiness
        if emotion == "happy": st.balloons()
        
        # 3D Interactive Donut Chart
        fig = px.pie(values=probs, names=encoder.classes_, hole=0.6, 
                     color_discrete_sequence=px.colors.sequential.Bluered)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🎧 Recorded Audio")
        st.audio(file_path)

# --- UI Interface ---
st.markdown('<h1 class="main-title">🎤 AI Voice Emotion Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:20px; color:#94a3b8;">Developed by <b>Aliza Mumtaz</b></p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎙️ Record Audio", "📂 Upload File"])

with tab1:
    if st.button("Start Recording (8s)"):
        rec = sd.rec(int(8 * 44100), samplerate=44100, channels=1)
        sd.wait()
        wavio.write("audio.wav", rec, 44100, sampwidth=2)
        process_and_predict("audio.wav")

with tab2:
    file = st.file_uploader("Upload your .wav audio file", type=["wav"])
    if file:
        with open("upload.wav", "wb") as f: f.write(file.getbuffer())
        process_and_predict("upload.wav")

# --- Footer ---
st.markdown("<br><hr><p style='text-align:center; color:gray;'>© 2026 Aliza Mumtaz | AI Voice Analysis Project</p>", unsafe_allow_html=True)