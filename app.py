import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown
import base64
from io import BytesIO

# Set page config
st.set_page_config(page_title="Skin Cancer Detective", page_icon="🎗️")

# Define the model filename
model_file = "skin_cancer_model.h5"

# Check if the model file exists; if not, download it from Google Drive
if not os.path.exists(model_file):
    st.info("Downloading AI model... Please wait ⏳")
    file_id = "1DPGyP60aUkKugxQ_XSLhtDATVq41U0_j"  # Replace with your actual file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_file, quiet=False)
    st.success("Download complete! ✅")

# Load the AI model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# App title and description
st.title("🎗️ Skin Cancer Detective")
st.write("Upload or take a photo of a skin lesion for AI-powered analysis.")

# Create tabs for the two input methods
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Take Photo"])

# Function to process image and make prediction
def process_image(img):
    with st.spinner("Analyzing image..."):
        img = img.convert('RGB')  # Ensure RGB format
        img_display = img.copy()
        img = img.resize((224, 224))
        st.image(img_display, caption="Your Skin Spot", use_column_width=True)
        
        # Predict
        img_array = np.array(img) / 255.0
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        
        class_names = ['Melanoma', 'Nevus', 'Basal Cell Carcinoma', 'Actinic Keratosis',
                      'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion']
        
        result = class_names[np.argmax(prediction)]
        confidence = float(prediction[0][np.argmax(prediction)]) * 100
        
        # Show prediction with confidence
        st.success(f"🔍 Prediction: {result} (Confidence: {confidence:.1f}%)")
        
        # Important disclaimer
        st.warning("⚠️ IMPORTANT: This is not medical advice. Please consult a dermatologist for proper diagnosis.")

# Tab 1: Upload Image
with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            process_image(img)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Tab 2: Take Photo
with tab2:
    st.write("Please allow camera access when prompted")
    picture = st.camera_input("Take a picture")
    if picture is not None:
        try:
            img = Image.open(picture)
            process_image(img)
        except Exception as e:
            st.error(f"Error processing camera image: {str(e)}")

# Add information section
with st.expander("About this app"):
    st.write("""
    This app uses a deep learning model to analyze skin lesion images.
    The model was trained on the HAM10000 dataset and can identify 7 different types of skin conditions.
    Always consult with a medical professional for proper diagnosis.
    """)
    
# Add footer
st.markdown("---")
st.markdown("Created for educational purposes only. Not for medical use.")
