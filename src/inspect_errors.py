
from pathlib import Path
import csv
import shutil
import numpy as np
import tensorflow as tf

# ==========================================
# 1. Project paths
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2_v3.keras"
)

VAL_DIR = PROJECT_DIR / "data" / "processed_v2" / "val"

OUTPUT_DIR = PROJECT_DIR / "models" / "misclassified_validation"
REPORT_PATH = PROJECT_DIR / "models" / "misclassified_validation.csv"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

# ==========================================
# 2. Check required paths
# ==========================================
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VAL_DIR.is_dir():
    raise FileNotFoundError(f"Validation dataset not found: {VAL_DIR}")

# ==========================================
# 3. Load validation images
# ==========================================
print("Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = val_ds.class_names

# Get the exact original file paths in dataset order
file_paths = val_ds.file_paths

# ==========================================
# 4. Load model and check class order
# ==========================================
print("\nLoading PlantGuardAI V3...")

model = tf.keras.models.load_model(MODEL_PATH)

train_dir = PROJECT_DIR / "data" / "processed_v2" / "train"

train_class_names = sorted(
    folder.name
    for folder in train_dir.iterdir()
    if folder.is_dir()
)

if class_names != train_class_names:
    raise ValueError(
        f"Class order mismatch.\n"
        f"Training: {train_class_names}\n"
        f"Validation: {class_names}"
    )

# ==========================================
# 5. Get actual labels and predictions
# ==========================================
print("\nPredicting validation images...")

y_true = np.concatenate([
    labels.numpy() for _, labels in val_ds
])

probabilities = model.predict(val_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)
confidence = np.max(probabilities, axis=1)

if len(file_paths) != len(y_true):
    raise ValueError("Image paths and labels do not match.")

# ==========================================
# 6. Find misclassified images
# ==========================================
wrong_indices = np.where(y_true != y_pred)[0]

# Remove previous output so results are from this run only
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 7. Save incorrect images and CSV report
# ==========================================
with open(
    REPORT_PATH, "w", newline="", encoding="utf-8"
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "image",
        "actual_class",
        "predicted_class",
        "prediction_confidence_percent"
    ])

    for index in wrong_indices:
        image_path = Path(file_paths[index])

        actual_class = class_names[int(y_true[index])]
        predicted_class = class_names[int(y_pred[index])]

        # Separate mistakes by actual and predicted class
        destination_dir = (
            OUTPUT_DIR / f"actual_{actual_class}__predicted_{predicted_class}"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir / image_path.name

        # Avoid overwriting files with duplicate names
        if destination_path.exists():
            destination_path = destination_dir / (
                f"{index}_{image_path.name}"
            )

        shutil.copy2(image_path, destination_path)

        writer.writerow([
            str(image_path),
            actual_class,
            predicted_class,
            round(float(confidence[index]) * 100, 2)
        ])

# ==========================================
# 8. Display summary
# ==========================================
total = len(y_true)
correct = total - len(wrong_indices)

print("\n" + "=" * 48)
print("PLANTGUARDAI V3 - VALIDATION ERROR ANALYSIS")
print("=" * 48)

print("Validation images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", len(wrong_indices))
print(f"Validation accuracy: {correct / total * 100:.2f}%")

print("\nMisclassification breakdown:")

for actual_index, actual_name in enumerate(class_names):
    for predicted_index, predicted_name in enumerate(class_names):
        if actual_index == predicted_index:
            continue

        count = int(np.sum(
            (y_true == actual_index) &
            (y_pred == predicted_index)
        ))

        print(
            f"{actual_name} -> {predicted_name}: {count}"
        )

print("\nIncorrect images saved to:", OUTPUT_DIR)
print("CSV report saved to:", REPORT_PATH)
print("\nError analysis completed.")
