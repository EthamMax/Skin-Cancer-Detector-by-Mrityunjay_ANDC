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
# Set up Streamlit Page for Full-Screen View
# ======================
st.set_page_config(
    page_title="AI Skin Cancer Detector",
    page_icon="🎗️",
    layout="wide",  # Full-screen layout
    initial_sidebar_state="collapsed"
)

# ======================
# Custom CSS for Full-Screen Effect & UI Enhancements
# ======================
st.markdown("""
    <style>
        /* Full screen */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 90%;
        }
        /* Center align text */
        .center-text {
            text-align: center;
        }
        /* Improve camera preview */
        .stCamera > div {
            display: flex;
            justify-content: center;
        }
        /* Improve image display */
        .stImage img {
            border-radius: 10px;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ======================
# App Introduction (Shown First)
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
st.write("Take a live photo using your camera or upload an existing image.")

col1, col2 = st.columns(2, gap="large")

# Upload Image
uploaded_file = col1.file_uploader("📁 Upload a skin lesion image", type=["jpg", "jpeg", "png"], help="Upload an image from your device.")

# Take Photo with Camera (With a Stylish Frame)
camera_img = col2.camera_input("📷 Capture an image using your camera")

# Assign Image from Either Upload or Camera
img = None
if uploaded_file:
    img = Image.open(uploaded_file)
elif camera_img:
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

    # Preprocess image
    img_array = np.array(img.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    with st.spinner("🩺 Analyzing image... ⏳"):
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds)
        confidence = np.max(preds) * 100

        # Define class labels
        class_names = ['Melanoma (Cancerous)', 'Benign (Non-Cancerous)']

    # Display Prediction
    with col4:
        st.subheader(f"Prediction: **{class_names[predicted_class]}**")
        st.metric(label="Confidence", value=f"{confidence:.2f}%", delta="AI Confidence Score")

    # ======================
    # Grad-CAM Heatmap
    # ======================
    st.subheader("🧐 How AI Sees Your Image (Grad-CAM)")
    last_conv_layer_name = "conv2d_4"  # Replace with your actual last conv layer
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(np.array(img.resize((224, 224))), 0.6, heatmap, 0.4, 0)

    col5, col6 = st.columns(2, gap="large")

    with col5:
        st.image(img, caption="📷 Original Image", use_column_width=True)
    with col6:
        st.image(superimposed_img, caption="📊 Grad-CAM Visualization", use_column_width=True)

    st.warning("⚠️ **Important:** This AI tool is for educational purposes only. It is not a substitute for medical diagnosis.")
