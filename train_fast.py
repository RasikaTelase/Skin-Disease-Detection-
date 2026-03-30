"""
Fast Training Script - Optimized for quick model training
Uses the split dataset (train/val/test) from data/split_data
"""

import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("=" * 80)
print("FAST SKIN DISEASE MODEL TRAINING")
print("=" * 80)

# Enable mixed precision for faster training
tf.config.optimizer.set_jit(True)

# --------------------------------------------------
# STEP 1: DATA SETUP
# --------------------------------------------------
print("\n[Step 1/5] Loading Dataset...")

# Data augmentation for training
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
        batch_size=batch_size,
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

# Get class info from directory structure
class_names = sorted([d for d in os.listdir(config.TRAIN_DIR) if os.path.isdir(os.path.join(config.TRAIN_DIR, d))])
num_classes = len(class_names)
class_indices = {i: name for i, name in enumerate(class_names)}

print(f"[OK] Classes: {num_classes}")
print(f"[OK] Found class names: {class_names}")

# Estimate samples from directory (faster than unbatch)
def count_images(directory):
    """Count images in directory"""
    return sum(len(files) for _, _, files in os.walk(directory))

train_samples = count_images(config.TRAIN_DIR)
val_samples = count_images(config.VAL_DIR)
test_samples = count_images(config.TEST_DIR)

print(f"[OK] Training samples: {train_samples}")
print(f"[OK] Validation samples: {val_samples}")
print(f"[OK] Test samples: {test_samples}")

# Save class indices
class_indices_path = config.CLASS_INDICES_PATH
os.makedirs(os.path.dirname(class_indices_path), exist_ok=True)
with open(class_indices_path, 'w') as f:
    json.dump(class_indices, f, indent=4)
print(f"[OK] Class indices saved: {class_indices_path}")

# --------------------------------------------------
# STEP 2: BUILD MODEL (ResNet50 - Fast & Efficient)
# --------------------------------------------------
print("\n[Step 2/5] Building Model Architecture...")

# Use ResNet50 for balance of speed and accuracy
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
)

# Freeze base model initially
base_model.trainable = False

# Build classification head
inputs = tf.keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[Step 2/5] Building Model Architecture...")

# Use ResNet50 for balance of speed and accuracy
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
)

# Freeze base model initially
base_model.trainable = False

# Build classification head
inputs = tf.keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"[OK] Model built successfully")
print(f"[OK] Total parameters: {model.count_params():,}")

# --------------------------------------------------
# STEP 3: FAST TRAINING (Stage 1)
# --------------------------------------------------
print("\n[Step 3/5] Stage 1 - Training Classification Head (Frozen Base)...")

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(config.CHECKPOINT_DIR, 'stage1_best.keras'),
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

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # Reduced for speed
    callbacks=callbacks,
    verbose=1
)

print("\n[Step 3/5] Stage 1 - Training Classification Head (Frozen Base)...")

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(config.CHECKPOINT_DIR, 'stage1_best.keras'),
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

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

print("[OK] Stage 1 training completed!")

# --------------------------------------------------
# STEP 4: FINE-TUNING (Stage 2)
# --------------------------------------------------
print("\n[Step 4/5] Stage 2 - Fine-tuning Selected Layers...")

# Unfreeze last 20 layers of ResNet50
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage2 = [
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

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks_stage2,
    verbose=1
)

print("[OK] Stage 2 fine-tuning completed!")

# --------------------------------------------------
# STEP 5: EVALUATION & SAVE
# --------------------------------------------------
print("\n[Step 5/5] Evaluating Model on Test Set...")

test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
print(f"\n[OK] Test Accuracy: {test_accuracy*100:.2f}%")
print(f"[OK] Test Loss: {test_loss:.4f}")

# Save final model
os.makedirs(config.MODEL_DIR, exist_ok=True)
model.save(config.MODEL_PATH)
print(f"[OK] Model saved: {config.MODEL_PATH}")

# Save training results
results = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "num_classes": num_classes,
    "total_training_samples": train_samples,
    "total_validation_samples": val_samples,
    "total_test_samples": test_samples,
    "model_parameters": model.count_params(),
    "stage1_epochs": 15,
    "stage2_epochs": 10
}

results_path = os.path.join(config.CHECKPOINT_DIR, 'training_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"[OK] Results saved: {results_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Model saved: {config.MODEL_PATH}")
print(f"   Class indices: {class_indices_path}")
print("\nNext Steps:")
print(f"   1. Use src/predict.py for inference")
print(f"   2. Run web_app/app.py for web interface")
print("=" * 80 + "\n")