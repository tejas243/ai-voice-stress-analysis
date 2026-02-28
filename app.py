from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

import plotly.graph_objects as go
import streamlit as st
import numpy as np
import librosa
import joblib
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import tempfile
import soundfile as sf

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Voice Stress Analysis", layout="centered")

# ------------------- LOGO -------------------
logo = Image.open("logo.jpg")

col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=80)

st.markdown("""
<h1 style='text-align: center; 
           background: linear-gradient(to right, #00BFFF, #8A2BE2);
           -webkit-background-clip: text;
           color: transparent;'>
🎙 AI Voice Stress Analysis System
</h1>
""", unsafe_allow_html=True)

st.write("Upload a .wav file or record live audio to analyze stress level using Machine Learning.")
st.markdown("---")
st.info("⚠️ This system analyzes vocal stress patterns and does not directly detect lies.")

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ------------------- FEATURE EXTRACTION -------------------
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)

    return np.hstack([mfcc, chroma, mel])

# ------------------- PDF GENERATION -------------------
def generate_pdf(prediction_label, probabilities, stress_score):
    filename = "AI_Stress_Report.pdf"
    doc = SimpleDocTemplate(filename)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Voice Stress Analysis Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"<b>Final Stress Level:</b> {prediction_label}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"<b>Overall Stress Index:</b> {round(stress_score,2)}%", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    data = [
        ["Stress Type", "Confidence (%)"],
        ["Low Stress", round(probabilities[0]*100,2)],
        ["Medium Stress", round(probabilities[1]*100,2)],
        ["High Stress", round(probabilities[2]*100,2)]
    ]

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN',(1,1),(-1,-1),'CENTER')
    ]))

    elements.append(table)
    doc.build(elements)

    return filename

# ------------------- AUDIO PROCESSOR -------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame):
        self.audio_frames.append(frame)
        return frame

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp.wav"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path, format="audio/wav")

    features = extract_features(temp_path)
    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    prediction_label = ["Low Stress", "Medium Stress", "High Stress"][prediction]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("Low Stress")
    elif prediction == 1:
        st.warning("Medium Stress")
    else:
        st.error("High Stress")

    stress_score = (
        probabilities[0] * 30 +
        probabilities[1] * 60 +
        probabilities[2] * 90
    )

    # ---- Bar Chart ----
    labels = ["Low Stress", "Medium Stress", "High Stress"]
    values = [probabilities[0]*100, probabilities[1]*100, probabilities[2]*100]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=labels,
        y=values,
        text=[f"{v:.2f}%" for v in values],
        textposition="auto",
        marker_color=["#2ecc71", "#f1c40f", "#e74c3c"]
    ))

    fig_bar.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Gauge ----
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stress_score,
        title={"text": "Overall Stress Index"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "#2ecc71"},
                {"range": [40, 70], "color": "#f1c40f"},
                {"range": [70, 100], "color": "#e74c3c"}
            ]
        }
    ))

    gauge_fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # ---- PDF ----
    pdf_file = generate_pdf(prediction_label, probabilities, stress_score)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download AI Stress Report (PDF)",
            data=f,
            file_name="AI_Stress_Report.pdf",
            mime="application/pdf"
        )

    os.remove(temp_path)

# ------------------- LIVE MIC -------------------
st.markdown("### 🎤 Or Record Live Audio")

webrtc_ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDRECV,  # 👈 Audio playback enabled
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx.audio_processor:

    if st.button("Process Recorded Audio"):

        frames = webrtc_ctx.audio_processor.audio_frames

        if len(frames) > 0:

            audio_data = np.concatenate(
                [frame.to_ndarray() for frame in frames],
                axis=1
            )

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)

            audio_data = audio_data.astype("float32")

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

            sf.write(temp_file.name, audio_data, 48000)

            features = extract_features(temp_file.name)
            features = scaler.transform([features])

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            prediction_label = ["Low Stress", "Medium Stress", "High Stress"][prediction]

            st.subheader("Prediction Result")

            if prediction == 0:
                st.success("Low Stress")
            elif prediction == 1:
                st.warning("Medium Stress")
            else:
                st.error("High Stress")

            stress_score = (
                probabilities[0] * 30 +
                probabilities[1] * 60 +
                probabilities[2] * 90
            )

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stress_score,
                title={"text": "Overall Stress Index"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 40], "color": "#2ecc71"},
                        {"range": [40, 70], "color": "#f1c40f"},
                        {"range": [70, 100], "color": "#e74c3c"}
                    ]
                }
            ))

            gauge_fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(gauge_fig, use_container_width=True)

            pdf_file = generate_pdf(prediction_label, probabilities, stress_score)

            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="📄 Download AI Stress Report (PDF)",
                    data=f,
                    file_name="AI_Stress_Report.pdf",
                    mime="application/pdf"
                )

            try:
                os.remove(temp_file.name)
            except:
                pass

        else:
            st.warning("No audio recorded yet.")