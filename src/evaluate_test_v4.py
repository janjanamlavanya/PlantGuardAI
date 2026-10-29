
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==========================================
# 1. Project paths
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "data" / "processed_v3" / "test"
MODEL_PATH = PROJECT_DIR / "models" / "plantguard_mobilenetv2_v4.keras"
RESULTS_DIR = PROJECT_DIR / "results"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

# ==========================================
# 2. Check required files
# ==========================================

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# ==========================================
# 3. Load test dataset
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = test_ds.class_names
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print("Classes:", class_names)

# ==========================================
# 4. Load trained model
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# 5. Generate predictions
# ==========================================

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# ==========================================
# 6. Calculate evaluation metrics
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 50)
print("PLANTGUARDAI V4 — FINAL TEST RESULTS")
print("=" * 50)

print(f"Test images: {len(y_true)}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0,
))

# ==========================================
# 7. Confusion matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)

print("Confusion Matrix:")
print(cm)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    values_format="d",
    xticks_rotation=25,
)

plt.title("PlantGuardAI V4 — Test Confusion Matrix")
plt.tight_layout()

figure_path = RESULTS_DIR / "v4_test_confusion_matrix.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nConfusion matrix saved to:", figure_path)
print("Test evaluation completed.")
