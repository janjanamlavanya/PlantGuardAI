
from pathlib import Path
import tensorflow as tf

# -----------------------------------
# 1. Project paths and settings
# -----------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "data" / "processed" / "train"
VAL_DIR = PROJECT_DIR / "data" / "processed" / "val"

INITIAL_MODEL = PROJECT_DIR / "models" / "plantguard_mobilenetv2.keras"
FINETUNED_MODEL = PROJECT_DIR / "models" / "plantguard_mobilenetv2_finetuned.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# -----------------------------------
# 2. Check required paths
# -----------------------------------
if not TRAIN_DIR.is_dir():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not VAL_DIR.is_dir():
    raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

if not INITIAL_MODEL.is_file():
    raise FileNotFoundError(f"Initial model not found: {INITIAL_MODEL}")

# -----------------------------------
# 3. Load datasets
# -----------------------------------
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
print("\nClasses:", class_names)

if len(class_names) != 2:
    raise ValueError(f"Expected two classes, found: {class_names}")

if class_names != val_ds.class_names:
    raise ValueError("Training and validation class order differs.")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------------
# 4. Load the existing trained model
# -----------------------------------
print("\nLoading the existing MobileNetV2 model...")

model = tf.keras.models.load_model(INITIAL_MODEL)

# Find the pretrained MobileNetV2 backbone
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError(
        "Could not find the MobileNetV2 backbone in the saved model."
    )

# -----------------------------------
# 5. Unfreeze selected later layers
# -----------------------------------
base_model.trainable = True

# Freeze most layers; fine-tune only the final layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen for stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print(
    "Trainable MobileNetV2 layers:",
    sum(layer.trainable for layer in base_model.layers)
)

# -----------------------------------
# 6. Compile with a very low learning rate
# -----------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------------
# 7. Save the best fine-tuned model
# -----------------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=FINETUNED_MODEL,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
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

# -----------------------------------
# 8. Fine-tune the model
# -----------------------------------
print("\nStarting MobileNetV2 fine-tuning...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# -----------------------------------
# 9. Evaluate the best saved model
# -----------------------------------
print("\nLoading the best fine-tuned model...")

best_model = tf.keras.models.load_model(FINETUNED_MODEL)

val_loss, val_accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print("\n--- PlantGuardAI Fine-Tuning Results ---")
print("Classes:", class_names)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print("Fine-tuned model saved to:", FINETUNED_MODEL)
