
from pathlib import Path
import random
import shutil

# ==========================================
# 1. Project paths and settings
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed_v2"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

random.seed(RANDOM_SEED)

# ==========================================
# 2. Check the original dataset
# ==========================================
if not RAW_DIR.is_dir():
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_DIR}"
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUTPUT_DIR}\n"
        "Check its contents before choosing another output folder."
    )

class_folders = sorted(
    folder for folder in RAW_DIR.iterdir()
    if folder.is_dir()
)

if len(class_folders) != 2:
    raise ValueError(
        f"Expected 2 class folders, found {len(class_folders)}."
    )

if not abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-9:
    raise ValueError("Split ratios must add up to 1.")

# ==========================================
# 3. Split each class separately
# ==========================================
total_train = 0
total_val = 0
total_test = 0

for class_folder in class_folders:

    images = [
        file for file in class_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Use relative paths to prevent filename collisions
    images.sort()
    random.shuffle(images)

    count = len(images)

    if count < 3:
        raise ValueError(
            f"Not enough images in {class_folder.name}"
        )

    train_end = int(count * TRAIN_RATIO)
    val_end = train_end + int(count * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    for split_name, split_images in splits.items():

        destination = (
            OUTPUT_DIR / split_name / class_folder.name
        )
        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for image_path in split_images:
            # Preserve subfolder structure and unique filenames
            relative_path = image_path.relative_to(class_folder)
            destination_file = destination / relative_path

            destination_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(image_path, destination_file)

    print(f"\nClass: {class_folder.name}")
    print(f"Total images: {count}")
    print(f"Training: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Testing: {len(test_images)}")

    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

# ==========================================
# 4. Display summary
# ==========================================
print("\n" + "=" * 45)
print("PLANTGUARDAI V2 DATASET PREPARATION")
print("=" * 45)

print("Training images:", total_train)
print("Validation images:", total_val)
print("Testing images:", total_test)
print("Total images:", total_train + total_val + total_test)
print("Dataset location:", OUTPUT_DIR)

print("\nDataset split completed successfully.")
