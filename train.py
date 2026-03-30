



import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

print("=" * 80)
print("SKIN DISEASE DETECTION - PROPER 2-STAGE TRAINING")
print("=" * 80)

# --------------------------------------------------
# STEP 1: DATA PREPARATION
# --------------------------------------------------
print("\n[Step 1/6] Setting Up Data Generators...")

# Proper data augmentation for medical images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    config.VAL_DIR,
    target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_data.num_classes
print(f"✓ Classes: {num_classes}")
print(f"✓ Training samples: {train_data.samples}")
print(f"✓ Validation samples: {val_data.samples}")

# Save class indices
class_indices_path = config.CLASS_INDICES_PATH
with open(class_indices_path, 'w') as f:
    json.dump(train_data.class_indices, f, indent=4)
print(f"✓ Class indices saved: {class_indices_path}")

# --------------------------------------------------
# STEP 2: MODEL ARCHITECTURE
# --------------------------------------------------
print("\n[Step 2/6] Building Model Architecture...")

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
)

# Freeze base model for Stage 1
base_model.trainable = False

# Build custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

print(f"✓ Model created")
print(f"✓ Total parameters: {model.count_params():,}")
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"✓ Trainable parameters: {trainable_params:,}")

# --------------------------------------------------
# STEP 3: STAGE 1 - TRAIN CLASSIFICATION HEAD
# --------------------------------------------------
print("\n[Step 3/6] STAGE 1: Training Classification Head (Base Frozen)...")
print(f"Learning Rate: 0.0001")
print(f"Epochs: 10 (Fast Training)")

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

callbacks_stage1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,  # Reduced for faster training
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
        min_lr=1e-7,
        verbose=1
    )
]

print("\nTraining Stage 1...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # Reduced for faster training
    callbacks=callbacks_stage1,
    verbose=1
)

stage1_acc = max(history1.history['val_accuracy'])
print(f"\n✓ Stage 1 Best Val Accuracy: {stage1_acc*100:.2f}%")

# --------------------------------------------------
# STEP 4: STAGE 2 - FINE-TUNE TOP LAYERS
# --------------------------------------------------
print("\n[Step 4/6] STAGE 2: Fine-Tuning Top ResNet Layers...")

# Unfreeze the top layers of ResNet50
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"✓ Trainable parameters: {trainable_params:,}")
print(f"Learning Rate: 0.00001 (10x lower)")
print(f"Epochs: 15 (Fast Training)")

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

callbacks_stage2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,  # Reduced for faster training
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
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

print("\nTraining Stage 2...")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # Reduced for faster training
    callbacks=callbacks_stage2,
    verbose=1
)

stage2_acc = max(history2.history['val_accuracy'])
print(f"\n✓ Stage 2 Best Val Accuracy: {stage2_acc*100:.2f}%")

# --------------------------------------------------
# STEP 5: SAVE FINAL MODEL
# --------------------------------------------------
print("\n[Step 5/6] Saving Final Model...")

model.save(config.MODEL_PATH)
print(f"✓ Model saved: {config.MODEL_PATH}")

# --------------------------------------------------
# STEP 6: FINAL EVALUATION
# ---------------------------------------------------+

print("\n[Step 6/6] Final Evaluation...")

results = model.evaluate(val_data, verbose=0)
final_loss = results[0]
final_acc = results[1]
final_top3 = results[2]

print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n📊 FINAL RESULTS:")
print(f"   Validation Loss: {final_loss:.4f}")
print(f"   Validation Accuracy: {final_acc*100:.2f}%")
print(f"   Top-3 Accuracy: {final_top3*100:.2f}%")

if final_acc > 0.60:
    print(f"\n✅ EXCELLENT: Model achieved {final_acc*100:.1f}% accuracy!")
elif final_acc > 0.40:
    print(f"\n✓ GOOD: Model achieved {final_acc*100:.1f}% accuracy!")
elif final_acc > 0.25:
    print(f"\n⚠️ MODERATE: Model achieved {final_acc*100:.1f}% accuracy - could be better")
else:
    print(f"\n❌ POOR: Model only achieved {final_acc*100:.1f}% accuracy - check data quality")

print(f"\n📁 Model Path: {config.MODEL_PATH}")
print(f"📁 Class Indices: {class_indices_path}")
print("\n✅ Ready for predictions!")
print("=" * 80)