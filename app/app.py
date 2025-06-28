import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from io import BytesIO
import base64

# Page Configuration
st.set_page_config(page_title="Pattern Sense", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: 600;
        color: white;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: white;
        margin-top: 4px;
    }
    .card {
        margin: 2rem auto;
        padding: 1.5rem;
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
        width: 360px;
        text-align: center;
    }
    .card img {
        border-radius: 12px;
        width: 180px;
        height: 180px;
        object-fit: cover;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 20px;
        color: #333;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    .confidence {
        font-size: 16px;
        color: #777;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Text ---
st.markdown("<div class='title'>🧵 Pattern Sense</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a fabric image and let the model identify its pattern!</div>", unsafe_allow_html=True)

# --- Load Model ---
MODEL_PATH = "outputs/fabric_classifier.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error("🚨 Model not found at outputs/fabric_classifier.h5")
    st.stop()

# --- Class Names (Ensure same order used during training) ---
class_names = ['Checks', 'Dotted', 'Floral', 'Geometric', 'Herringbone',
               'Ikat', 'Paisley', 'Plain', 'Printed', 'Striped']

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload a fabric image", type=["jpg", "jpeg", "png"])

# --- Image to Base64 ---
def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# --- On Upload ---
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    input_shape = model.input_shape[1:3]
    resized_img = img.resize(input_shape)

    # Preprocess
    img_array = image.img_to_array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    img_b64 = image_to_base64(img)

    # --- Display Prediction Card ---
    st.markdown(f"""
        <div class="card">
            <img src="data:image/png;base64,{img_b64}" alt="Uploaded Image"/>
            <div class="result-text">Predicted Pattern: <strong>{predicted_class}</strong></div>
            <div class="confidence">Confidence: {confidence:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)
