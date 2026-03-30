"""
Resume Training Script - Continue from saved checkpoint at Epoch 34
Resumes from stage2_best.keras checkpoint
"""

import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("=" * 80)
print("RESUMING TRAINING FROM EPOCH 34")
print("=" * 80)

# Enable mixed precision
tf.config.optimizer.set_jit(True)

# --------------------------------------------------
# STEP 1: DATA SETUP
# --------------------------------------------------
print("\n[Step 1/4] Loading Dataset...")

def create_augmented_dataset(directory, batch_size, augment=True):
    """Load dataset with optional augmentation"""
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
    else:
        data_augmentation = None
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        seed=42,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        shuffle=augment
    )
    
    # Normalize
    dataset = dataset.map(lambda x, y: (x / 255.0, y))
    
    # Apply augmentation if provided
    if data_augmentation:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Load training data with augmentation
train_data = create_augmented_dataset(
    config.TRAIN_DIR,
    batch_size=config.BATCH_SIZE,
    augment=True
)

# Load validation data without augmentation
val_data = create_augmented_dataset(
    config.VAL_DIR,
    batch_size=config.BATCH_SIZE,
    augment=False
)

# Load test data without augmentation
test_data = create_augmented_dataset(
    config.TEST_DIR,
    batch_size=config.BATCH_SIZE,
    augment=False
)

print("[OK] Datasets loaded")

# Get class info
class_names = sorted([d for d in os.listdir(config.TRAIN_DIR) if os.path.isdir(os.path.join(config.TRAIN_DIR, d))])
num_classes = len(class_names)

print(f"[OK] Number of classes: {num_classes}")

# --------------------------------------------------
# STEP 2: LOAD SAVED MODEL FROM CHECKPOINT
# --------------------------------------------------
print("\n[Step 2/4] Loading Checkpoint from Epoch 34...")

checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'stage2_best.keras')
if os.path.exists(checkpoint_path):
    print(f"[OK] Found checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path)
    print(f"[OK] Model loaded successfully")
    print(f"[OK] Total parameters: {model.count_params():,}")
else:
    print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
    print(f"[ERROR] Looked in: {config.CHECKPOINT_DIR}")
    sys.exit(1)

# --------------------------------------------------
# STEP 3: RESUME TRAINING FROM EPOCH 34
# --------------------------------------------------
print("\n[Step 3/4] Resuming Training from Epoch 34...")
print("[OK] Will train for up to 26 more epochs (target: 60)")

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(config.CHECKPOINT_DIR, 'stage2_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# Resume training starting from epoch 34
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=60,
    initial_epoch=34,
    callbacks=callbacks,
    verbose=1
)

print("[OK] Training completed!")

# --------------------------------------------------
# STEP 4: EVALUATION & SAVE
# --------------------------------------------------
print("\n[Step 4/4] Evaluating Model on Test Set...")

test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
print(f"\n[OK] Test Accuracy: {test_accuracy*100:.2f}%")
print(f"[OK] Test Loss: {test_loss:.4f}")

# Save final model
os.makedirs(config.MODEL_DIR, exist_ok=True)
model.save(config.MODEL_PATH)
print(f"[OK] Model saved: {config.MODEL_PATH}")

# Save results
results = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "num_classes": num_classes,
    "model_parameters": model.count_params(),
    "training_type": "Resumed from Epoch 34",
    "total_epochs": 60,
    "resumed_from_epoch": 34
}

results_path = os.path.join(config.CHECKPOINT_DIR, 'training_results_resumed.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"[OK] Results saved: {results_path}")

print("\n" + "=" * 80)
print("RESUMED TRAINING COMPLETE!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Model saved: {config.MODEL_PATH}")
print(f"   Resumed from epoch: 34")
print(f"   Total epochs: 60")
print("=" * 80 + "\n")