import os
os.system("pip install gdown==4.7.1")
import gdown

import os
import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

"""
AI-Powered Skin Cancer Detection Web App with Grad-CAM Visualization
Developed by: Mrityunjay Kumar
Acharya Narendra Dev College, University of Delhi

This web app uses a pre-trained deep learning model to detect skin cancer
from uploaded skin lesion images and applies Grad-CAM to visualize the regions
of interest. This project is for educational purposes only and is not a substitute
for professional medical diagnosis.
"""

# Set Streamlit page configuration
st.set_page_config(page_title="Skin Cancer Detection 🩺", page_icon="🎗️")

# Define the model filename (use the same name everywhere)
model_file = "skin_cancer_model.h5"

# Check if the model file exists; if not, download it from Google Drive
if not os.path.exists(model_file):
    st.info("Downloading AI model... Please wait ⏳")
    file_id = "1DPGyP60aUkKugxQ_XSLhtDATVq41U0_j"  # Replace with your actual file ID from Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_file, quiet=False)
    st.success("Download complete! ✅")

# Load the AI model with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_file)
    # Force the model to build its computational graph with a dummy input
    dummy_input = np.zeros((1, 224, 224, 3))
    _ = model.predict(dummy_input)
    return model

model = load_model()

# Define the Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    
    # Compute gradients of the predicted class with respect to the conv layer output
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map by its corresponding gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize the heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

# App title and description
st.title("🎗️ Skin Cancer Detective")
st.write("Upload a skin lesion image to see the AI prediction and its Grad-CAM visualization.")

# Let users upload images
uploaded_file = st.file_uploader("Upload a skin spot photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open the image and ensure it's in RGB mode
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Resize image to 224x224 for the model
    image_resized = image.resize((224, 224))
    
    # Preprocess the image for prediction (normalize and add batch dimension)
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get prediction from the model
    prediction = model.predict(img_array)
    class_names = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma',
                   'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']
    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]
    confidence = prediction[0][pred_index] * 100
    st.success(f"🔍 Prediction: {predicted_class} (Confidence: {confidence:.1f}%)")
    
    # Generate and display Grad-CAM visualization
    # Set the last convolutional layer name (update if needed based on your model's architecture)
    last_conv_layer_name = "conv2d"  # Example layer name; adjust if necessary
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the original image
    img_array_orig = np.array(image_resized)
    superimposed_img = cv2.addWeighted(img_array_orig, 0.6, heatmap, 0.4, 0)
    st.image(superimposed_img, caption="Grad-CAM Visualization", use_column_width=True)
    
    # Display disclaimer
    st.warning("⚠️ This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")
