from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

print("PlantGuardAI - Dataset Checker")
print("-" * 35)

total_images = 0

for class_folder in sorted(DATASET_DIR.iterdir()):
    if class_folder.is_dir():
        count = sum(
            1 for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_folder.name}: {count} images")
        total_images += count

print("-" * 35)
print(f"Total images: {total_images}")