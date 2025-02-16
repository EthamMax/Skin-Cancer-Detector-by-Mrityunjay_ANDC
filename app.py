import streamlit as st
import numpy as np
import tensorflow as tf
import os
import requests
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image

# ======================
# Set Full-Screen Page Layout
# ======================
st.set_page_config(
    page_title="AI Skin Cancer Detector",
    page_icon="🎗️",
    layout="wide",  # Full-screen layout
    initial_sidebar_state="collapsed"
)

# ======================
# Custom CSS for Centered UI & Improved Styling
# ======================
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 80%; margin: auto; }
        .center-text { text-align: center; }
        .stButton>button { width: 100%; height: 50px; font-size: 16px; font-weight: bold; background-color: #2E86C1; color: white; border-radius: 10px; }
        .stImage img { border-radius: 10px; box-shadow: 3px 3px 10px rgba(0,0,0,0.2); }
    </style>
""", unsafe_allow_html=True)

# ======================
# App Introduction (Centered)
# ======================
st.markdown("""
    <h1 class='center-text' style='color: #2E86C1;'>
        🔬 AI-Powered Skin Cancer Detection with Grad-CAM
    </h1>
    <h3 class='center-text'>Developed by: <b>Mrityunjay Kumar</b></h3>
    <h4 class='center-text'>Acharya Narendra Dev College, University of Delhi</h4>
    <div style='text-align: justify; margin: 20px 0;'>
        <p>Skin cancer is one of the most common cancers worldwide, and early detection 
        can save lives. This AI-powered tool analyzes skin lesions to provide a preliminary 
        diagnosis using deep learning.</p>
        <p><b>Why use Grad-CAM?</b></p>
        <ul>
            <li><b>Visual AI Explanations</b> – See where the AI model is focusing.</li>
            <li><b>More Transparency</b> – Understand how the model makes decisions.</li>
            <li><b>Medical Assistance</b> – AI can assist dermatologists for early diagnosis.</li>
        </ul>
        <p>⚠️ <b>Disclaimer:</b> This web app is for <b>educational</b> purposes only. 
        It is NOT a substitute for a professional medical diagnosis.</p>
    </div>
""", unsafe_allow_html=True)

st.divider()  # Adds a visual separator

# ======================
# Load Model with Download Progress Bar
# ======================
model_url = "https://huggingface.co/MrityuTron/skin-cancer-detector-by-mrityunjay-kumar-andc/resolve/main/skin_cancer_model.h5"
model_file = "skin_cancer_model.h5"

def download_model():
    """Downloads the model from Hugging Face with a progress bar."""
    if not os.path.exists(model_file):
        st.info("Downloading AI model from Hugging Face... ⏳")

        with st.progress(0, text="Starting download...") as progress_bar:
            start_time = time.time()
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192  # 8 KB
            downloaded_size = 0

            with open(model_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = downloaded_size / total_size
                    elapsed_time = time.time() - start_time
                    speed = downloaded_size / (elapsed_time + 1e-8) / 1_000_000  # MB/s
                    progress_bar.progress(progress, text=f"Downloaded {downloaded_size//1_000_000}MB / {total_size//1_000_000}MB ({speed:.2f} MB/s)")

            st.success("Download complete! ✅")

download_model()

@st.cache_resource
def load_model():
    """Loads the TensorFlow model"""
    return tf.keras.models.load_model(model_file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ======================
# Image Upload / Capture Section (Centered)
# ======================
st.header("📸 Capture or Upload an Image")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    img = None
    option = st.radio("Choose an option:", ("📁 Upload Image", "📷 Take Photo"), horizontal=True)

    if option == "📁 Upload Image":
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)

    elif option == "📷 Take Photo":
        camera_img = st.camera_input("")
        if camera_img:
            img = Image.open(camera_img)

# ======================
# Prediction & Grad-CAM (Only Runs After an Image is Uploaded)
# ======================
if img:
    st.divider()
    st.subheader("🔬 AI Skin Cancer Analysis")

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.image(img, caption="📷 Input Image", use_column_width=True)

    # Convert image to RGB
    img = img.convert("RGB")

    # Resize & preprocess image
    img_array = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Make prediction
    with st.spinner("🩺 Analyzing image... ⏳"):
        try:
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds)
            confidence = np.max(preds) * 100

            # Ensure class names match the model's output
            class_names = ['Melanoma', 'Melanocytic Nevus', 'Basal Cell Carcinoma',
                           'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion']

            # Debugging - Print Model Output
            st.write(f"🔍 Raw Model Output: {preds}")

            # Check for out-of-range index error
            if predicted_class >= len(class_names):
                st.error(f"⚠️ Prediction Error: Index {predicted_class} is out of range! The model might be incorrect.")
                st.stop()  # Stop execution if error

            # Display Prediction
            with col4:
                st.subheader(f"Prediction: **{class_names[predicted_class]}**")
                st.metric(label="Confidence", value=f"{confidence:.2f}%", delta="AI Confidence Score")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.stop()

    st.warning("⚠️ **Important Notice:** This AI-powered tool is for educational purposes only. It should not be used as a substitute for professional medical diagnosis.")
