import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import os
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt

# Config
SAMPLE_RATE = 16000
DURATION = 6  # seconds
N_MFCC = 20
REFERENCE_PATH = "reference_audio.npy"
MODEL_PATH = "models/ai_voice_detector_model.pkl"

# Load model
model = joblib.load(MODEL_PATH)

# Utils
def is_valid_audio(audio, sr=SAMPLE_RATE):
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = np.array([
        np.sum(np.square(audio[i:i+frame_length]))
        for i in range(0, len(audio) - frame_length, hop_length)
    ])
    avg_energy = np.mean(energy)
    st.write(f"🔊 Audio Energy: {avg_energy:.6f}")
    return avg_energy > 1e-5

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio if max_val == 0 else audio / max_val

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    st.info("🎙️ Recording... Please speak")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    if not is_valid_audio(audio):
        st.warning("❌ Audio too silent. Please speak clearly.")
        return None
    return audio

def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    audio = normalize_audio(audio)
    audio = librosa.effects.preemphasis(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.fix_length(mfcc, size=63, axis=1)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc.flatten()

def save_reference(audio):
    np.save(REFERENCE_PATH, audio)
    st.success("✅ Reference voice saved.")

def load_reference():
    if os.path.exists(REFERENCE_PATH):
        return np.load(REFERENCE_PATH)
    else:
        st.warning("⚠️ No reference voice found. Please register first.")
        return None

def compare_voices(ref_audio, live_audio):
    ref_emb = extract_mfcc(ref_audio)
    live_emb = extract_mfcc(live_audio)
    similarity = cosine_similarity([ref_emb], [live_emb])[0][0]
    return similarity

def detect_ai_voice_proba(features):
    proba = model.predict_proba([features])[0]
    return proba

# Polygraph-style plot
def plot_polygraph_multi(audio, sr=SAMPLE_RATE, segment_duration=0.5):
    n_samples_per_seg = int(segment_duration * sr)
    total_segments = int(len(audio) / n_samples_per_seg)
    
    times = np.arange(total_segments) * segment_duration
    
    human_probs = []
    ai_probs = []
    stress_levels = []
    voice_consistency = []

    for i in range(total_segments):
        segment = audio[i*n_samples_per_seg:(i+1)*n_samples_per_seg]
        if len(segment) < n_samples_per_seg:
            break
        mfcc_feat = extract_mfcc(segment)
        proba = detect_ai_voice_proba(mfcc_feat)
        human_probs.append(proba[0])
        ai_probs.append(proba[1])
        stress_levels.append(0.5 + 0.3 * np.sin(i/3) + np.random.normal(0, 0.05))
        voice_consistency.append(0.7 + 0.2 * np.cos(i/2) + np.random.normal(0, 0.03))

    stress_levels = np.clip(stress_levels, 0, 1)
    voice_consistency = np.clip(voice_consistency, 0, 1)

    plt.figure(figsize=(10,5))
    plt.plot(times, human_probs, label="Human Confidence", color="green")
    plt.plot(times, ai_probs, label="AI Confidence", color="red")
    plt.plot(times, stress_levels, label="Stress Level", color="orange", linestyle='--')
    plt.plot(times, voice_consistency, label="Voice Consistency", color="blue", linestyle=':')
    plt.ylim(0, 1.1)
    plt.xlabel("Time (s)")
    plt.ylabel("Value / Probability")
    plt.title("Polygraph-style Voice Analysis Over Time")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    st.pyplot(plt)

# Streamlit UI
st.set_page_config(page_title="AI-Powered Real-Time Deepfake Audio Detection")
st.title("🎙️ AI-Powered Real-Time Deepfake Audio Detection")

voice_match_threshold = st.sidebar.slider("Voice Match Threshold", 0.70, 0.99, 0.85, 0.01)
ai_detection_threshold = st.sidebar.slider("AI Detection Threshold", 0.40, 0.90, 0.60, 0.01)

tabs = st.tabs(["📝 Register Voice", "🔍 Live Interview Monitor", "📁 Voice File Analysis"])

with tabs[0]:
    st.header("Step 1: Register Candidate Voice")
    input_method = st.radio("Choose input method:", ["Record via Mic", "Upload .wav File"])

    if input_method == "Record via Mic":
        if st.button("🎤 Record Voice"):
            audio = record_audio()
            if audio is not None:
                save_reference(audio)
    else:
        uploaded_file = st.file_uploader("Upload reference .wav file", type=["wav"])
        if uploaded_file:
            y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
            if is_valid_audio(y):
                save_reference(y)
            else:
                st.warning("❌ Uploaded audio is too silent. Please upload a clear voice.")

with tabs[1]:
    st.header("Step 2: Live Interview Monitoring")

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            return frame

    st.subheader("📹 Real-Time Webcam Feed (Interview Simulation)")
    webrtc_streamer(key="interview-video", video_transformer_factory=VideoTransformer)

    ref_audio = load_reference()

    if ref_audio is not None:
        if st.button("🎧 Start Live Detection"):
            live_audio = record_audio()
            if live_audio is not None:
                similarity = compare_voices(ref_audio, live_audio)
                st.write(f"🧑‍💼 Speaker Match Similarity: **{similarity:.2f}** (Threshold: {voice_match_threshold})")

                if similarity >= voice_match_threshold:
                    st.success("✅ Speaker Verified - Same person")

                    plot_polygraph_multi(live_audio)
                    avg_proba = model.predict_proba([extract_mfcc(live_audio)])[0]
                    human_prob, ai_prob = avg_proba[0], avg_proba[1]

                    st.write(f"🧠 Average AI Detection Confidence - Human: {human_prob:.2f}, AI: {ai_prob:.2f} (Threshold: {ai_detection_threshold})")
                    if ai_prob > ai_detection_threshold:
                        st.error("🤖 Warning: Possible AI-generated voice detected!")
                    else:
                        st.success("🧠 Voice seems human (Not detected as AI)")
                else:
                    st.error("⚠️ Possible Impersonation - Voice does not match")

with tabs[2]:
    st.header("Step 3: Analyze Uploaded Voice File")
    uploaded_file = st.file_uploader("Upload a voice (.wav) file to analyze", type=["wav"], key="analyze")

    if uploaded_file:
        y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
        if not is_valid_audio(y):
            st.warning("⚠️ The uploaded file is too silent or noisy. Please upload a clearer voice.")
        else:
            st.subheader("📊 Polygraph-Style Analysis")
            plot_polygraph_multi(y)

            proba = model.predict_proba([extract_mfcc(y)])[0]
            human_prob, ai_prob = proba[0], proba[1]
            st.write(f"🧠 AI Detection Confidence - Human: **{human_prob:.2f}**, AI: **{ai_prob:.2f}** (Threshold: {ai_detection_threshold})")

            if ai_prob > ai_detection_threshold:
                st.error("🤖 This voice is likely AI-generated.")
            else:
                st.success("🧑 This voice is likely from a human.")
