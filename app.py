import streamlit as st
import tensorflow as tf
import os
import requests

# ======================
# Set up Streamlit Page
# ======================
st.set_page_config(page_title="AI Skin Cancer Detection", page_icon="🎗️")

# Hugging Face Model URL (Replace with your actual Hugging Face URL)
model_url = "https://huggingface.co/MrityuTron/skin-cancer-detector-by-mrityunjay-kumar-andc/resolve/main/skin_cancer_model.h5"
model_file = "skin_cancer_model.h5"

def download_model():
    """Downloads the model from Hugging Face if it does not exist locally."""
    if not os.path.exists(model_file):
        st.info("Downloading AI model from Hugging Face... ⏳")
        response = requests.get(model_url, stream=True)
        with open(model_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Download complete! ✅")

# Download the model before loading it
download_model()

@st.cache_resource
def load_model():
    """Loads the TensorFlow model"""
    return tf.keras.models.load_model(model_file)

# Try loading the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
