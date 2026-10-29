
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# 1. Project settings
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2_v4.keras"
)

IMAGE_SIZE = (160, 160)

CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___healthy",
]

DISPLAY_NAMES = {
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___healthy": "Healthy Tomato Leaf",
}

# ==========================================
# 2. Page configuration
# ==========================================

st.set_page_config(
    page_title="PlantGuardAI",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 PlantGuardAI")
st.subheader("AI-Powered Tomato Leaf Classification")

st.write(
    "Upload a tomato leaf image to classify it as "
    "Early Blight or Healthy using a trained MobileNetV2 model."
)

st.info(
    "This demo recognizes only two tomato leaf classes. "
    "It is an educational tool, not a substitute for expert diagnosis."
)

# ==========================================
# 3. Load the trained model
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# ==========================================
# 4. Upload image
# ==========================================

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"],
)

# ==========================================
# 5. Predict image class
# ==========================================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded tomato leaf",
            use_container_width=True,
        )

        if st.button("Analyze Leaf", type="primary"):
            with st.spinner("Analyzing image..."):

                resized_image = image.resize(IMAGE_SIZE)

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32,
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0,
                )

                probabilities = model.predict(
                    image_array,
                    verbose=0,
                )[0]

                if len(probabilities) != len(CLASS_NAMES):
                    st.error("Unexpected model output.")
                    st.stop()

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(
                    probabilities[predicted_index]
                )

            st.divider()
            st.header("Analysis Results")

            st.success(
                f"Prediction: {DISPLAY_NAMES[predicted_class]}"
            )

            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%",
            )

            st.write("Class probabilities")

            for index, class_name in enumerate(CLASS_NAMES):
                probability = float(probabilities[index])

                st.write(DISPLAY_NAMES[class_name])
                st.progress(
                    max(0.0, min(1.0, probability))
                )
                st.caption(f"{probability * 100:.2f}%")

            st.caption(
                "Confidence represents the model's predicted "
                "probability, not a guarantee of diagnosis."
            )

    except Exception as error:
        st.error(f"Unable to process this image: {error}")

st.divider()
st.caption("PlantGuardAI | Deep Learning Project | MobileNetV2")
