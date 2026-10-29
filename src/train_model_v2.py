
from pathlib import Path
import numpy as np
import tensorflow as tf

# ==========================================
# 1. Project paths and settings
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "data" / "processed_v3" / "train"
VAL_DIR = PROJECT_DIR / "data" / "processed_v3" / "val"

MODEL_PATH = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2_v4.keras"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Check dataset folders
# ==========================================

for folder in (TRAIN_DIR, VAL_DIR):
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder missing: {folder}")

# ==========================================
# 3. Load training and validation datasets
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

if num_classes != 2:
    raise ValueError(
        f"Expected 2 classes, found {num_classes}: {class_names}"
    )

if class_names != val_ds.class_names:
    raise ValueError("Training and validation classes do not match.")

print("\nClasses:", class_names)

# ==========================================
# 4. Calculate class weights
# ==========================================

class_counts = np.zeros(num_classes, dtype=np.int64)

for _, labels in train_ds:
    class_counts += np.bincount(
        labels.numpy(),
        minlength=num_classes
    )

if np.any(class_counts == 0):
    raise ValueError("At least one class has no training images.")

total = int(class_counts.sum())

class_weights = {
    i: total / (num_classes * int(class_counts[i]))
    for i in range(num_classes)
}

print("\nTraining image counts:")
for i, name in enumerate(class_names):
    print(f"{name}: {class_counts[i]}")

print("\nClass weights:")
for i, name in enumerate(class_names):
    print(f"{name}: {class_weights[i]:.3f}")

# ==========================================
# 5. Optimize input pipeline
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# 6. Data augmentation
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation")

# ==========================================
# 7. Load pretrained MobileNetV2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ==========================================
# 8. Build classification model
# ==========================================

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

x = augmentation(inputs)

x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# 9. Callbacks
# ==========================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ==========================================
# 10. Phase 1: Train classification head
# ==========================================

print("\nPHASE 1: Training classification head")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 11. Phase 2: Fine-tune MobileNetV2
# ==========================================

print("\nPHASE 2: Fine-tuning MobileNetV2")

base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 12. Load best model and evaluate validation
# ==========================================

print("\nLoading best saved model...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Training did not save a model at: {MODEL_PATH}"
    )

best_model = tf.keras.models.load_model(MODEL_PATH)

val_loss, val_accuracy = best_model.evaluate(val_ds, verbose=1)

# ==========================================
# 13. Display results
# ==========================================

print("\n" + "=" * 45)
print("PLANTGUARDAI V4 TRAINING RESULTS")
print("=" * 45)

print("Classes:", class_names)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print("Saved model:", MODEL_PATH)
print("\nTraining completed successfully.")
