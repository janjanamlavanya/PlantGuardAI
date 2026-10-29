
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. Project paths and settings
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2_v3.keras"
)

TEST_DIR = PROJECT_DIR / "data" / "processed_v2" / "test"

RESULTS_DIR = PROJECT_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

# ==========================================
# 2. Check required files and folders
# ==========================================
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not TEST_DIR.is_dir():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_DIR}"
    )

# ==========================================
# 3. Load the independent test dataset
# ==========================================
print("Loading independent test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names

if len(class_names) != 2:
    raise ValueError(
        f"Expected 2 classes, found {len(class_names)}"
    )

# ==========================================
# 4. Load the trained V3 model
# ==========================================
print("\nLoading PlantGuardAI V3 model...")

model = tf.keras.models.load_model(MODEL_PATH)

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        "Model output count does not match test classes."
    )

# Verify the model's output class order against
# the class order used during training.
train_dir = PROJECT_DIR / "data" / "processed_v2" / "train"

train_class_names = sorted(
    folder.name
    for folder in train_dir.iterdir()
    if folder.is_dir()
)

if class_names != train_class_names:
    raise ValueError(
        f"Class order mismatch!\n"
        f"Training: {train_class_names}\n"
        f"Testing: {class_names}"
    )

# ==========================================
# 5. Collect true labels and predictions
# ==========================================
print("\nPredicting test images...")

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

probabilities = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

# ==========================================
# 6. Calculate evaluation metrics
# ==========================================
accuracy = accuracy_score(y_true, y_pred)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

# ==========================================
# 7. Print results
# ==========================================
print("\n" + "=" * 50)
print("PLANTGUARDAI V3 - INDEPENDENT TEST RESULTS")
print("=" * 50)

print("Model:", MODEL_PATH.name)
print("Classes:", class_names)
print("Test images:", len(y_true))
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(report_text)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# 8. Save confusion matrix
# ==========================================
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("PlantGuardAI V3 - Test Confusion Matrix")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

matrix_path = RESULTS_DIR / "confusion_matrix_v3_test.png"

plt.savefig(
    matrix_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 9. Save metrics to a JSON report
# ==========================================
results = {
    "model": MODEL_PATH.name,
    "dataset": "data/processed_v2/test",
    "test_images": int(len(y_true)),
    "classes": class_names,
    "accuracy": float(accuracy),
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist()
}

report_path = RESULTS_DIR / "test_results_v3.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 50)
print("TEST EVALUATION COMPLETED")
print("=" * 50)
print("Confusion matrix:", matrix_path)
print("Metrics report:", report_path)
