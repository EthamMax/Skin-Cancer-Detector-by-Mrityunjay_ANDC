import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ======================
# Set up Streamlit Page
# ======================
st.set_page_config(page_title="AI Skin Cancer Detection", page_icon="🎗️")

st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        AI-Powered Skin Cancer Detection Web App with Grad-CAM Visualization
    </h1>
    <h3 style='text-align: center;'>Developed by: Mrityunjay Kumar</h3>
    <h4 style='text-align: center;'>Acharya Narendra Dev College, University of Delhi</h4>
    <div style='text-align: center; margin: 20px 0;'>
        <em>This web app uses a pre-trained deep learning model to detect skin cancer 
        from uploaded skin lesion images and applies Grad-CAM to visualize the regions
        of interest. This project is for educational purposes only and is not a substitute
        for professional medical diagnosis.</em>
    </div>
""", unsafe_allow_html=True)

# ======================
# Load Model (Download if missing)
# ======================
model_file = "skin_cancer_model.h5"
file_id = "1DPGyP60aUkKugxQ_XSLhtDATVq41U0_j"  # Google Drive File ID

if not os.path.exists(model_file):
    st.info("Downloading AI model... Please wait ⏳")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_file, quiet=False)
    st.success("Download complete! ✅")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_file)

model = load_model()

# ======================
# Image Upload / Capture
# ======================
st.divider()
option = st.radio("Choose input method:", ("Upload Image", "Take Photo from Camera"), horizontal=True)

img = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
elif option == "Take Photo from Camera":
    camera_img = st.camera_input("Take a photo of skin lesion")
    if camera_img:
        img = Image.open(camera_img)

# ======================
# Prediction & Grad-CAM
# ======================
if img:
    st.divider()
    st.subheader("Analysis Results")

    # Preprocess image
    img_array = np.array(img.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    with st.spinner("Analyzing image..."):
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds)
        confidence = np.max(preds) * 100

        # Update with your model's classes
        class_names = ['Melanoma', 'Benign']  

        # Grad-CAM Implementation
        last_conv_layer_name = "conv2d_4"  # Update with your last convolutional layer name
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, predicted_class]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap.numpy(), (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(np.array(img.resize((224, 224))), 0.6, heatmap, 0.4, 0)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
        st.write(f"**Prediction:** {class_names[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    with col2:
        st.image(superimposed_img, caption="Grad-CAM Visualization", use_column_width=True)

    st.divider()
    st.warning("**Important Notice:** This AI-powered tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.")
