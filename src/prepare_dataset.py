
from pathlib import Path
import random
import shutil

# Project and dataset paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Dataset settings
TRAIN_RATIO = 0.80
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(RANDOM_SEED)

if not RAW_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {RAW_DIR}")

# Find the two class folders
class_folders = sorted(
    folder for folder in RAW_DIR.iterdir()
    if folder.is_dir()
)

if len(class_folders) != 2:
    raise ValueError(
        f"Expected 2 class folders, found {len(class_folders)}. "
        "Check data/raw."
    )

# Prevent accidentally mixing with an earlier split
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. "
        "Rename or remove it only after checking its contents."
    )

total_train = 0
total_val = 0

for class_folder in class_folders:
    images = [
        file for file in class_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) < 2:
        raise ValueError(
            f"Not enough images in {class_folder.name}"
        )

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create destination folders
    train_dir = OUTPUT_DIR / "train" / class_folder.name
    val_dir = OUTPUT_DIR / "val" / class_folder.name

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Copy images; keep originals unchanged
    for image_path in train_images:
        shutil.copy2(image_path, train_dir / image_path.name)

    for image_path in val_images:
        shutil.copy2(image_path, val_dir / image_path.name)

    print(f"\nClass: {class_folder.name}")
    print(f"Original images: {len(images)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    total_train += len(train_images)
    total_val += len(val_images)

print("\n" + "-" * 40)
print(f"Total training images: {total_train}")
print(f"Total validation images: {total_val}")
print(f"Total images prepared: {total_train + total_val}")
print(f"\nDataset saved to: {OUTPUT_DIR}")
