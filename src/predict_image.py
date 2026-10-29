
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

# ==========================================
# 1. Project paths and settings
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2_v4.keras"
)

TEST_IMAGE = PROJECT_DIR / "test_images" / "tomato_test.JPG"

IMAGE_SIZE = (160, 160)

CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___healthy",
]

# ==========================================
# 2. Check required files
# ==========================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if len(sys.argv) > 1:
    IMAGE_PATH = Path(sys.argv[1]).expanduser()
    if not IMAGE_PATH.is_absolute():
        IMAGE_PATH = PROJECT_DIR / IMAGE_PATH
else:
    IMAGE_PATH = TEST_IMAGE

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Place an image in test_images or provide its path."
    )

# ==========================================
# 3. Load model and image
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
)

image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

# ==========================================
# 4. Predict
# ==========================================

probabilities = model.predict(image_array, verbose=0)[0]

if len(probabilities) != len(CLASS_NAMES):
    raise ValueError(
        "Model output count does not match the configured class names."
    )

predicted_index = int(np.argmax(probabilities))
predicted_class = CLASS_NAMES[predicted_index]
confidence = float(probabilities[predicted_index]) * 100

# ==========================================
# 5. Display results
# ==========================================

print("\n" + "=" * 45)
print("PLANTGUARDAI V4 — IMAGE PREDICTION")
print("=" * 45)

print(f"Image: {IMAGE_PATH.name}")
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

print("\nClass probabilities:")
for name, probability in zip(CLASS_NAMES, probabilities):
    print(f"{name}: {probability * 100:.2f}%")

print("\nPrediction completed.")
