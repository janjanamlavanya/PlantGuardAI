
from pathlib import Path
import csv
import random
import math
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent

REPORT_PATH = PROJECT_DIR / "models" / "misclassified_validation.csv"

NUM_IMAGES = 20
SEED = 42

if not REPORT_PATH.is_file():
    raise FileNotFoundError(
        f"Error report not found: {REPORT_PATH}"
    )

# Read the saved error report
with open(REPORT_PATH, newline="", encoding="utf-8") as file:
    errors = list(csv.DictReader(file))

if not errors:
    print("No misclassified validation images found!")
    raise SystemExit

# Select up to 20 examples reproducibly
random.seed(SEED)
samples = random.sample(errors, min(NUM_IMAGES, len(errors)))

columns = 4
rows = math.ceil(len(samples) / columns)

fig, axes = plt.subplots(
    rows, columns,
    figsize=(16, 4 * rows)
)

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for ax, item in zip(axes, samples):
    image_path = Path(item["image"])

    if not image_path.is_file():
        ax.set_title("Image not found")
        ax.axis("off")
        continue

    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))

    actual = item["actual_class"].replace("Tomato___", "")
    predicted = item["predicted_class"].replace("Tomato___", "")
    confidence = item["prediction_confidence_percent"]

    ax.set_title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence}%",
        fontsize=9
    )
    ax.axis("off")

# Hide unused grid cells
for ax in axes[len(samples):]:
    ax.axis("off")

plt.suptitle(
    "PlantGuardAI V3 - Misclassified Validation Images",
    fontsize=15
)
plt.tight_layout()
plt.show()
