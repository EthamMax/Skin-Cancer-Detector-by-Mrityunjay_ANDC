# SkinVision AI: Skin Cancer Detection Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)  [// Replace with your Streamlit Sharing or Hugging Face Spaces app link once deployed]

## Overview

**AISkin Cancer Detector** is a web application built using Streamlit and TensorFlow that leverages the power of Artificial Intelligence to assist in skin cancer detection. By uploading a dermoscopic image of a skin lesion, users can receive an AI-powered assessment of the likelihood of melanoma and a Grad-CAM visualization highlighting the areas the AI focused on for its diagnosis.

**Key Features:**

*   **AI-Powered Skin Lesion Analysis:** Utilizes a deep learning model (MobileNetV2) fine-tuned on the HAM10000 dataset to classify skin lesions into 7 categories, including melanoma.
*   **Grad-CAM Visualization:** Provides visual explanations of AI's decision-making process using Grad-CAM heatmaps, helping users understand which parts of the image were most important for the diagnosis.
*   **Interactive User Interface:** Built with Streamlit for a user-friendly and interactive experience. Users can upload images from their local storage or take live photos using their device camera.
*   **Early Melanoma Detection Focus:** Aims to assist in the early detection of melanoma, the deadliest form of skin cancer, emphasizing the importance of timely diagnosis.

**Disclaimer:**

**This web application is for educational and demonstration purposes only.** It is a proof-of-concept project and **NOT intended to be a substitute for professional medical advice, diagnosis, or treatment.** Always consult with a qualified dermatologist or healthcare professional for any concerns about skin health or potential skin cancer.

## Technologies Used:

*   **Python:** Programming language
*   **TensorFlow/Keras:** Deep learning framework for building and training the AI model
*   **Streamlit:** Python library for creating the web application
*   **tf-explain:** Library for Grad-CAM visualization
*   **NumPy, Pandas, OpenCV, PIL:** Data processing and image handling libraries

## Dataset:

*   **HAM10000 dataset:**  [https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-2018](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-2018)

## Model Architecture:

*   Transfer Learning with MobileNetV2

## Getting Started:

Instructions on how to run your app locally (if applicable) or access the deployed app (once deployed).

## Project Creator:

*   **Mrityunjay Kumar** - Biomedical Science Student, Acharya Narendra Dev College, University of Delhi. 

## License:

[Choose a license, e.g., MIT License, Apache 2.0, or leave as "No License" if you don't want to specify a license]

---

**Feel free to expand on this template, add more details about your project, model performance, challenges you faced, and how you overcame them. A good README is crucial for showcasing your work!**

**Step 6.5: Download Files from Colab**

Now, you need to download the following files from your Colab notebook to your local computer so you can upload them to GitHub and Hugging Face:

*   `app.py`
*   `requirements.txt`
*   `README.md`
*   Your trained model weights file: `best_model.weights.h5` (from `/content/drive/MyDrive/skin_cancer_detection_model_checkpoints/best_model.weights.h5`)

**To download files from Colab:**

1.  **Go to the "Files" pane** in Colab (left sidebar).
2.  **For each file (`app.py`, `requirements.txt`, `README.md`):**
    *   Right-click on the file name.
    *   Select "**Download**". The file will be downloaded to your computer's default download location.
3.  **For `best_model.weights.h5`:**
    *   Navigate to the `drive/MyDrive/skin_cancer_detection_model_checkpoints` folder in the "Files" pane.
    *   Right-click on `best_model.weights.h5`.
    *   Select "**Download**".

**Step 6.6: Upload Files to GitHub**

1.  **Go to your GitHub repository** (e.g., `SkinCancerAI-WebApp`) in your browser.
2.  Click on the "**Add file**" button (usually a dropdown button that says "Add file") and select "**Upload files**".
3.  Drag and drop the files you downloaded from Colab (`app.py`, `requirements.txt`, `README.md`) into the GitHub upload area, or click "choose your files" to select them from your computer.
4.  Once the files are uploaded, click on the "**Commit changes**" button to finalize the upload to your GitHub repository.

**Step 6.7: Upload Model Weights to Hugging Face Hub**

1.  **Go to your Hugging Face Hub model repository** (e.g., `MrityunjayKumar/SkinCancerAI-Model`) in your browser.
2.  Click on the "**Files and versions**" tab (if not already selected).
3.  Click on the "**Add file**" button and select "**Upload files**".
4.  Drag and drop your `best_model.weights.h5` file into the Hugging Face upload area, or click "choose your files".
5.  You can add a commit message (optional). Click on the "**Commit changes**" button to upload the model weights file to your Hugging Face Hub repository.

**After completing these steps, your project code will be on GitHub, and your trained model weights will be on Hugging Face Hub! Your project is now shareable and ready for the world!** 🌍

Let me know when you have completed these upload steps, or if you encounter any issues. We are almost at the finish line! 🎉 Then, we can think about deployment and sharing your amazing "SkinVision AI" web app!
