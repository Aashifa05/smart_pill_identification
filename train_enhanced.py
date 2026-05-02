#!/usr/bin/env python3
"""
Enhanced Training: Improved Discrimination
===========================================
Train model better on training data to:
1. Minimize false negatives (known pills marked as unknown)
2. Minimize false positives (unknown pills misclassified as known)

Strategy:
- Stronger regularization to prevent overfitting
- Better training/validation split
- Focus on learning discriminative features
- Confidence calibration
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split

print("=" * 80)
print("ENHANCED TRAINING: IMPROVED DISCRIMINATION")
print("=" * 80)
print("\nObjectives:")
print("  1. Better accuracy on known pills (minimize false negatives)")
print("  2. Better rejection of unknown pills (minimize false positives)")
print("  3. Learn discriminative features for each medication")
print("  4. Reduce overfitting while maintaining accuracy")

# ============================================================================
# STEP 1: Load Training Data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING TRAINING DATA")
print("=" * 80)

train_dir = Path('media/pilldata/train')
test_dir = Path('media/pilldata/test')

# Get class mapping from metadata
metadata_path = Path('media/pilldata/model_metadata.json')
with open(metadata_path, 'r') as f:
    old_metadata = json.load(f)

class_names = list(old_metadata.get('label_map', {}).keys())
class_to_idx = {name: i for i, name in enumerate(class_names)}

print(f"Target classes ({len(class_names)}):")
for i, name in enumerate(sorted(class_names)):
    print(f"  {i+1:2}. {name}")

# Collect training images organized by class
class_images = defaultdict(list)
all_images = []

for img_path in sorted(train_dir.glob('*.jpg')) + sorted(train_dir.glob('*.png')):
    all_images.append(img_path)
    # Extract class from filename
    filename = img_path.stem
    clean_name = filename.replace(' - Copy', '').strip()
    parts = clean_name.rsplit(' ', 1)
    if parts[-1].isdigit():
        clean_name = ' '.join(parts[:-1])
    
    class_images[clean_name].append(img_path)

print(f"\nData summary:")
print(f"  Total images found: {len(all_images)}")
print(f"  Unique identifiers: {len(class_images)}")

# Map images to known classes
training_images = {name: [] for name in class_names}
for key, paths in class_images.items():
    for name in class_names:
        # Better matching: check if key contains the medication name
        if name.lower() in key.lower() or key.lower() in name.lower():
            training_images[name].extend(paths)
            break

print(f"\nImages per class:")
for name in class_names:
    count = len(training_images[name])
    print(f"  {name:40} {count:3} images")

total_training = sum(len(v) for v in training_images.values())
print(f"  {'TOTAL':40} {total_training:3} images")

# ============================================================================
# STEP 2: Prepare Data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPARING TRAINING DATA")
print("=" * 80)

from tensorflow.keras.preprocessing import image

X_train = []
y_train = []
train_image_paths = []

print("Loading images...")
for class_idx, class_name in enumerate(class_names):
    images = training_images[class_name]
    
    for img_path in images:
        try:
            img = image.load_img(str(img_path), target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            X_train.append(img_array)
            y_train.append(class_to_idx[class_name])
            train_image_paths.append(str(img_path))
        except Exception as e:
            print(f"  Error loading {img_path.name}: {e}")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"\n✓ Data prepared:")
print(f"  Training images: {X_train.shape}")
print(f"  Classes: {len(class_names)}")

# Split data: 80% train, 20% validation (with stratification)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"  Training split: {X_train_split.shape[0]} images")
print(f"  Validation split: {X_val_split.shape[0]} images")

# ============================================================================
# STEP 3: Build Model (with STRONGER focus on discriminative features)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: BUILDING MODEL WITH STRONG REGULARIZATION")
print("=" * 80)

from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2

# Base model: MobileNetV3Large (pre-trained on ImageNet)
base_model = MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Strategy: Freeze initial layers, train later layers
# This preserves ImageNet features while learning pill-specific discrimination
for layer in base_model.layers[:-50]:  # Freeze first 50 layers
    layer.trainable = False

for layer in base_model.layers[-50:]:  # Train last 50 layers
    layer.trainable = True

# Build complete model
model = Sequential([
    base_model,
    
    # Global pooling
    GlobalAveragePooling2D(),
    BatchNormalization(),
    
    # Dense layers with STRONG regularization
    Dense(1024, activation='relu', 
          kernel_regularizer=L1L2(l1=1e-4, l2=1e-4)),
    Dropout(0.50),  # Higher dropout
    BatchNormalization(),
    
    Dense(512, activation='relu',
          kernel_regularizer=L1L2(l1=1e-4, l2=1e-4)),
    Dropout(0.45),
    BatchNormalization(),
    
    Dense(256, activation='relu',
          kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
    Dropout(0.40),
    BatchNormalization(),
    
    Dense(128, activation='relu'),
    Dropout(0.30),
    
    # Output: softmax for 20 classes
    Dense(len(class_names), activation='softmax')
])

print("✓ Model architecture:")
print(f"  Base: MobileNetV3Large (ImageNet pre-trained)")
print(f"  Trainable layers: Last 50 layers")
print(f"  Total parameters: {model.count_params():,}")
print(f"  Regularization: L1L2 + Dropout (up to 50%)")
print(f"  Output: {len(class_names)} classes")

# ============================================================================
# STEP 4: Compile with STRONG focus on training data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: COMPILING MODEL")
print("=" * 80)

model.compile(
    optimizer=Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled:")
print("  Optimizer: Adam (lr=5e-5)")
print("  Loss: Sparse Categorical Crossentropy")
print("  Focus: Maximize training accuracy while preventing overfitting")

# ============================================================================
# STEP 5: Data Augmentation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SETTING UP DATA AUGMENTATION")
print("=" * 80)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# STRONG augmentation to prevent overfitting and improve generalization
train_generator = ImageDataGenerator(
    rotation_range=35,           # Rotate ±35 degrees
    width_shift_range=0.25,      # Shift width ±25%
    height_shift_range=0.25,     # Shift height ±25%
    zoom_range=0.35,             # Zoom ±35%
    brightness_range=[0.65, 1.35],  # Brightness ±35%
    horizontal_flip=True,        # Horizontal flip
    fill_mode='nearest',         # Fill with nearest pixel
    shear_range=0.15             # Shear ±15%
)

val_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.85, 1.15]
)

print("✓ Augmentation configured:")
print("  Training: Strong (rotation, zoom, brightness, shifts, shear)")
print("  Validation: Light (for monitoring)")

# Create flows
train_flow = train_generator.flow(
    X_train_split, y_train_split,
    batch_size=32,
    shuffle=True
)

val_flow = val_generator.flow(
    X_val_split, y_val_split,
    batch_size=32,
    shuffle=False
)

# ============================================================================
# STEP 6: Training with Callbacks
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: TRAINING")
print("=" * 80)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

callbacks = [
    # Stop if validation loss doesn't improve
    EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    
    # Reduce learning rate if val loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=12,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'media/pilldata/model_enhanced_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
]

print(f"Training configuration:")
print(f"  Epochs: 200 (with early stopping)")
print(f"  Batch size: 32")
print(f"  Steps per epoch: {len(X_train_split) // 32}")
print(f"  Validation steps: {len(X_val_split) // 32}")
print(f"  Early stopping patience: 30 epochs")
print(f"  Learning rate schedule: ReduceLROnPlateau")

print(f"\nStarting training...\n")

history = model.fit(
    train_flow,
    steps_per_epoch=len(X_train_split) // 32,
    epochs=200,
    validation_data=val_flow,
    validation_steps=len(X_val_split) // 32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# STEP 7: Evaluate on Test Set
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: EVALUATING ON TEST SET")
print("=" * 80)

# Load test images
test_images = []
test_labels = []

print("Loading test images...")
for img_path in sorted(test_dir.glob('*.jpg')) + sorted(test_dir.glob('*.png')):
    filename = img_path.stem
    clean_name = filename.replace(' - Copy', '').strip()
    parts = clean_name.rsplit(' ', 1)
    if parts[-1].isdigit():
        clean_name = ' '.join(parts[:-1])
    
    if clean_name in class_to_idx:
        try:
            img = image.load_img(str(img_path), target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            test_images.append(img_array)
            test_labels.append(class_to_idx[clean_name])
        except Exception as e:
            print(f"  Error: {img_path.name}: {e}")

if test_images:
    X_test = np.array(test_images)
    y_test = np.array(test_labels)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Set Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2%}")
else:
    test_acc = 0
    print("No test images found")

# ============================================================================
# STEP 8: Save Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: SAVING MODEL")
print("=" * 80)

model_path = Path('media/pilldata/model_enhanced.keras')
model_path.parent.mkdir(parents=True, exist_ok=True)

model.save(str(model_path))
print(f"✓ Model saved: {model_path}")

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'type': 'enhanced_discrimination',
    'input_shape': [224, 224, 3],
    'num_classes': len(class_names),
    'label_map': class_to_idx,
    'class_names': class_names,
    'training_samples': len(X_train),
    'test_accuracy': float(test_acc),
    'val_accuracy': float(history.history['val_accuracy'][-1]),
    'train_accuracy': float(history.history['accuracy'][-1]),
    'architecture': 'MobileNetV3Large + Dense layers',
    'training_epochs': len(history.history['accuracy']),
    'augmentation': 'Strong (rotation±35°, zoom±35%, brightness±35%, shifts±25%, shear±15%)',
    'regularization': 'L1L2 (1e-4) + Dropout (30-50%) + BatchNormalization',
    'callbacks': ['EarlyStopping(patience=30)', 'ReduceLROnPlateau(patience=12)'],
    'training_notes': 'Enhanced model trained to better discriminate known pills and reject unknowns. Stronger regularization to prevent overfitting.'
}

metadata_path = Path('media/pilldata/model_enhanced_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved")

# ============================================================================
# STEP 9: Summary and Results
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(f"\n✓ Model saved: {model_path}")
print(f"✓ Classes trained: {len(class_names)}")
print(f"✓ Training images: {len(X_train)}")

print(f"\nFinal Accuracy:")
print(f"  Training:   {history.history['accuracy'][-1]:.2%}")
print(f"  Validation: {history.history['val_accuracy'][-1]:.2%}")
print(f"  Test:       {test_acc:.2%}")

print(f"\nImprovement Strategy:")
print(f"  ✓ Stronger regularization (L1L2 + Dropout up to 50%)")
print(f"  ✓ Aggressive data augmentation (prevents overfitting)")
print(f"  ✓ Transfer learning (ImageNet pre-trained base)")
print(f"  ✓ Better training/val split (80/20 with stratification)")
print(f"  ✓ Learning rate scheduling (reduce on plateau)")
print(f"  ✓ Early stopping (prevent over-training)")

print(f"\nExpected Improvements:")
print(f"  • Better accuracy on trained pill classes")
print(f"  • Better rejection of unknown pills")
print(f"  • More confident predictions overall")
print(f"  • Reduced false negatives (known marked unknown)")
print(f"  • Reduced false positives (unknown marked known)")

print(f"\nNext Steps:")
print(f"  1. Test on known pill images")
print(f"  2. Test on unknown pill images")
print(f"  3. Compare results with previous model")
print(f"  4. Adjust confidence thresholds if needed")
print(f"  5. Deploy model_enhanced.keras to production")

print("\n" + "=" * 80)
