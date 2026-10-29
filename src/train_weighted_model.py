
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. Project paths and settings
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "data" / "processed" / "train"
VAL_DIR = PROJECT_DIR / "data" / "processed" / "val"

MODEL_PATH = (
    PROJECT_DIR / "models" / "plantguard_mobilenetv2_weighted.keras"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 2. Load datasets
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
print("Classes:", class_names)

if len(class_names) != 2:
    raise ValueError(f"Expected two classes, found {class_names}")

# Count training examples for each class.
class_counts = np.zeros(len(class_names), dtype=np.int64)

for _, labels in train_ds:
    class_counts += np.bincount(
        labels.numpy(),
        minlength=len(class_names)
    )

# Calculate balanced class weights.
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.repeat(
        np.arange(len(class_names)),
        class_counts
    )
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(weights)
}

print("Training class counts:", class_counts.tolist())
print("Class weights:", class_weights)

# ==========================================
# 3. Load pretrained MobileNetV2
# ==========================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ==========================================
# 4. Build the classifier
# ==========================================
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation")

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
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# 5. Compile
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. Training callbacks
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
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# ==========================================
# 7. Train using class weights
# ==========================================
print("\nStarting class-weighted training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 8. Evaluate best model
# ==========================================
print("\nLoading best weighted model...")

best_model = tf.keras.models.load_model(MODEL_PATH)

val_loss, val_accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print("\n--- PlantGuardAI Weighted Model Results ---")
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print("Model saved to:", MODEL_PATH)
