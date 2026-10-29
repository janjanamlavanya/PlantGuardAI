

from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. Project paths and settings
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_MODEL_PATH = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2.keras"
)

FINETUNED_MODEL_PATH = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2_finetuned.keras"
)
VAL_DIR = PROJECT_DIR / "data" / "processed" / "val"
RESULTS_DIR = PROJECT_DIR / "models"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

# Early Blight is class index 0 in your current dataset
BLIGHT_CLASS_INDEX = 0

# ==========================================
# 2. Check paths
# ==========================================
if not ORIGINAL_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Original model not found: {ORIGINAL_MODEL_PATH}"
    )

if not FINETUNED_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Fine-tuned model not found: {FINETUNED_MODEL_PATH}"
    )

if not VAL_DIR.is_dir():
    raise FileNotFoundError(
        f"Validation folder not found: {VAL_DIR}"
    )

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. Load validation dataset
# ==========================================
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = val_ds.class_names

print("\nClasses:", class_names)

if len(class_names) != 2:
    raise ValueError(
        f"Expected 2 classes, found {len(class_names)}"
    )

# Find Early Blight class index automatically
blight_matches = [
    i for i, name in enumerate(class_names)
    if "early_blight" in name.lower()
]

if len(blight_matches) != 1:
    raise ValueError(
        f"Could not identify Early Blight class. "
        f"Classes found: {class_names}"
    )

BLIGHT_CLASS_INDEX = blight_matches[0]

print("Early Blight class index:", BLIGHT_CLASS_INDEX)

# ==========================================
# 4. Load trained model
# ==========================================
print("\nLoading original MobileNetV2...")
original_model = tf.keras.models.load_model(
    ORIGINAL_MODEL_PATH
)

print("Loading fine-tuned MobileNetV2...")
finetuned_model = tf.keras.models.load_model(
    FINETUNED_MODEL_PATH
)

# ==========================================
# 5. Collect true labels
# ==========================================
y_true = np.concatenate([
    labels.numpy() for _, labels in val_ds
])

# ==========================================
# 6. Compare both models
# ==========================================
weighted_model_path = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2_weighted.keras"
)

if not weighted_model_path.is_file():
    raise FileNotFoundError(
        f"Weighted model not found: {weighted_model_path}"
    )

print("\nLoading class-weighted MobileNetV2...")
weighted_model = tf.keras.models.load_model(weighted_model_path)

models_to_compare = {
    "Original MobileNetV2": original_model,
    "Fine-tuned MobileNetV2": finetuned_model,
    "Class-weighted MobileNetV2": weighted_model
}

comparison_results = []

for model_name, model in models_to_compare.items():

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    probabilities = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        pos_label=BLIGHT_CLASS_INDEX,
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        pos_label=BLIGHT_CLASS_INDEX,
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        pos_label=BLIGHT_CLASS_INDEX,
        zero_division=0
    )

    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(f"Early Blight precision: {precision:.4f}")
    print(f"Early Blight recall: {recall:.4f}")
    print(f"Early Blight F1-score: {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    matrix_path = RESULTS_DIR / f"{safe_name}_confusion_matrix.png"

    plt.savefig(matrix_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved to: {matrix_path}")

    comparison_results.append({
        "Model": model_name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Early Blight Precision": round(precision, 4),
        "Early Blight Recall": round(recall, 4),
        "Early Blight F1": round(f1, 4)
    })

# ==========================================
# 7. Final comparison
# ==========================================
print("\n" + "=" * 65)
print("PLANTGUARDAI MODEL COMPARISON")
print("=" * 65)

for result in comparison_results:
    print(result)

print("\nEvaluation complete.")
print("Use validation results for comparison only.")
print("An independent test set is needed for final assessment.")