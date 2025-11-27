import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (256, 256)

MODEL_PATHS = {
    "MobileNetV2": "MobileNetV2_final.keras",
    "EfficientNetB0": "EfficientNetB0_final.keras"
}

# UPDATE ACCORDING TO YOUR DATASET
class_names = ["Healthy", "Leaf Rust", "Septoria", "Stripe Rust"]


# -----------------------------
# Load models only once
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    for model_name, path in MODEL_PATHS.items():
        models[model_name] = tf.keras.models.load_model(path)
    return models

models = load_models()


# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------
# Predict using all models
# -----------------------------
def predict_all(image_array):
    predictions = {}
    for model_name, model in models.items():
        preds = model.predict(image_array)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        predictions[model_name] = (class_names[class_id], round(confidence * 100, 2))
    return predictions


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="TreatAI", page_icon="🌾", layout="wide")

st.title("🌾 TreatAI – Multi-Model Plant Disease Detection")
st.write("Upload a wheat leaf image to detect diseases using **MobileNetV2** and **EfficientNetB0**.")

uploaded_img = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    image = Image.open(uploaded_img)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess(image)
    results = predict_all(img_array)

    st.subheader("🔍 Prediction Results")
    cols = st.columns(len(results))

    for col, (model_name, (pred_class, conf)) in zip(cols, results.items()):
        with col:
            st.markdown(f"### 🧠 {model_name}")
            st.markdown(f"**Prediction:** {pred_class}")
            st.markdown(f"**Confidence:** `{conf}%`")
