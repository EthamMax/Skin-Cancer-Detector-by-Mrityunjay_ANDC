import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Set page config first
st.set_page_config(page_title="Skin Cancer Detection 🩺", page_icon="🎗️")

# App title and description
st.title("Skin Cancer Detection 🩺")
st.write("Upload an image or take a photo using your camera to detect skin cancer.")

# Option to upload or take a photo
option = st.radio("Choose an option:", ("Upload an image", "Take a photo from camera"))

# Initialize image variable
image = None

if option == "Upload an image":
    # File uploader for image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

elif option == "Take a photo from camera":
    # Camera input for taking a photo
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        # Read the image
        image = Image.open(camera_image)
        st.image(image, caption="Captured Photo", use_column_width=True)

# If an image is available, process it
if image is not None:
    # Convert PIL image to numpy array for processing
    image_np = np.array(image)

    # Example: Convert the image to grayscale (replace with your actual processing logic)
    processed_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Display the processed image
    st.write("Processed Image (Grayscale):")
    st.image(processed_image, caption="Processed Image", use_column_width=True)

    # Add your skin cancer detection logic here
    # For example, you can use a pre-trained model to predict the result
    # prediction = model.predict(processed_image)
    # st.write(f"Prediction: {prediction}")
