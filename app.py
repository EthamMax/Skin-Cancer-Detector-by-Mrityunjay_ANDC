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
    layout="wide",
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

st.divider()

# ======================
# Model Loading (UPDATED)
# ======================
@st.cache_resource
def load_cancer_model():
    """Download and load pre-trained skin cancer classification model"""
    model_url = "https://huggingface.co/MrityuTron/skin-cancer-detector-by-mrityunjay-kumar-andc/resolve/main/skin_cancer_model.h5"
    model_path = "skin_cancer_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading AI model... This may take a minute."):
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
            else:
                st.error("❌ Failed to download model. Please check the link or try again later.")
                return None

    try:
        model = tf.keras.models.load_model(model_path)
        if model.output_shape[1] != 7:
            st.error("❌ Model architecture mismatch! Expected 7 output classes.")
            return None
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# ======================
# Class Labels (FIXED)
# ======================
CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanocytic Nevus',
    'Melanoma',
    'Vascular Lesion'
]

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
# Prediction Logic (FIXED)
# ======================
if img:
    st.divider()
    st.subheader("🔬 AI Skin Cancer Analysis")

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.image(img, caption="📷 Input Image", use_column_width=True)

    img = img.convert("RGB")
    img_array = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with st.spinner("🩺 Analyzing image... ⏳"):
        try:
            model = load_cancer_model()
            if model is None:
                st.stop()

            preds = model.predict(img_array)
            if np.sum(preds) > 1.0:
                preds = tf.nn.softmax(preds).numpy()

            confidence = np.max(preds) * 100
            predicted_class = np.argmax(preds)
            diagnosis = CLASS_NAMES[predicted_class]

            with col4:
                st.subheader(f"Prediction: **{diagnosis}**")
                st.metric(label="Confidence", value=f"{confidence:.2f}%", delta="AI Confidence Score")

            st.write("Raw Predictions:", preds)
            st.write("Class Probabilities:", dict(zip(CLASS_NAMES, preds[0])))
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.stop()

    st.warning("⚠️ **Important Notice:** This AI-powered tool is for educational purposes only. It should not be used as a substitute for professional medical diagnosis.")
