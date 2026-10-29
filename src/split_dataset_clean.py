
from pathlib import Path
import random
import shutil

# ==========================================
# 1. Project paths and settings
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw_clean"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed_v3"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==========================================
# 2. Validate settings and source folders
# ==========================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Train, validation, and test ratios must total 1.")

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_DIR}\n"
        "Complete the raw_clean dataset copy first."
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists.\n"
        "No files were changed. Rename or inspect it before continuing."
    )

classes = sorted(
    folder.name
    for folder in RAW_DIR.iterdir()
    if folder.is_dir()
)

expected_classes = {
    "Tomato___Early_blight",
    "Tomato___healthy",
}

if set(classes) != expected_classes:
    raise ValueError(
        f"Expected exactly these two classes: {sorted(expected_classes)}\n"
        f"Found: {classes}"
    )

# ==========================================
# 3. Prepare reproducible splitting
# ==========================================

random.seed(SEED)
plan = {}

for class_name in classes:
    class_dir = RAW_DIR / class_name

    images = sorted(
        file for file in class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No images found in {class_dir}")

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    plan[class_name] = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

# ==========================================
# 4. Copy images into separate splits
# ==========================================

try:
    for class_name, splits in plan.items():
        for split_name, images in splits.items():
            target_dir = OUTPUT_DIR / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for image_path in images:
                shutil.copy2(image_path, target_dir / image_path.name)

    print("\nDataset splitting completed successfully!")
    print(f"Output folder: {OUTPUT_DIR}")

    for split_name in ("train", "val", "test"):
        print(f"\n{split_name.upper()}")

        for class_name in classes:
            count = len(plan[class_name][split_name])
            print(f"  {class_name}: {count} images")

except Exception:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    raise
