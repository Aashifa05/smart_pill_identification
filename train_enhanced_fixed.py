"""
ENHANCED MODEL TRAINING - FIXED VERSION
Fixes the learning rate and regularization issues from previous attempt
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import json

# Set memory growth to avoid GPU issues
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("=" * 80)
print("ENHANCED TRAINING - FIXED VERSION")
print("=" * 80)
print("\nFixes from previous attempt:")
print("  1. Higher learning rate (5e-4 instead of 5e-5)")
print("  2. Moderate regularization (balanced dropout)")
print("  3. More trainable layers (unfreeze more)")
print("  4. Proper layer freezing strategy")
print()

# ============================================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING TRAINING DATA")
print("=" * 80)

data_dir = 'media/pilldata/train'  # Images are in train subdirectory

# Define 20 target classes
class_names = [
    'Amoxicillin 500 MG',
    'Atomoxetine 25 MG',
    'Calcitriol 0.00025 MG',
    'Oseltamivir 45 MG',
    'Ramipril 5 MG',
    'apixaban 2.5 MG',
    'aprepitant 80 MG',
    'benzonatate 100 MG',
    'carvedilol 3.125 MG',
    'celecoxib 200 MG',
    'duloxetine 30 MG',
    'eltrombopag 25 MG',
    'montelukast 10 MG',
    'mycophenolate mofetil 250 MG',
    'pantoprazole 40 MG',
    'pitavastatin 1 MG',
    'prasugrel 10 MG',
    'saxagliptin 5 MG',
    'sitagliptin 50 MG',
    'tadalafil 5 MG'
]

print(f"\nTarget classes ({len(class_names)}):")
for i, cls in enumerate(class_names, 1):
    print(f"   {i:2}. {cls}")

# Load images
images = []
labels = []
class_counts = {cls: 0 for cls in class_names}

print("\nLoading images...")
for filename in os.listdir(data_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Match filename to class
    matched_class = None
    for cls_name in class_names:
        if cls_name.lower() in filename.lower() or filename.lower().replace('_', ' ') in cls_name.lower():
            matched_class = cls_name
            break
    
    if matched_class is None:
        continue
    
    try:
        img_path = os.path.join(data_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img.resize((224, 224)))
        images.append(img)
        labels.append(class_names.index(matched_class))
        class_counts[matched_class] += 1
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

images = np.array(images, dtype=np.float32) / 255.0
labels = np.array(labels, dtype=np.int32)

print(f"\nData summary:")
print(f"  Total images found: {len(images)}")
print(f"\nImages per class:")
for cls_name in class_names:
    count = class_counts[cls_name]
    print(f"  {cls_name:40} {count:3} images")
print(f"  {'TOTAL':40} {sum(class_counts.values()):3} images")

if len(images) < 100:
    print("\n❌ ERROR: Not enough images loaded!")
    exit(1)

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPARING TRAINING DATA")
print("=" * 80)

# Split into train and val
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\n✓ Data prepared:")
print(f"  Training images: {X_train.shape}")
print(f"  Validation images: {X_val.shape}")
print(f"  Classes: {len(class_names)}")

# ============================================================================
# STEP 3: BUILD MODEL - BETTER ARCHITECTURE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: BUILDING MODEL - OPTIMIZED ARCHITECTURE")
print("=" * 80)

# Use MobileNetV3Large as base
base_model = keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Unfreeze more layers (last 100 instead of 50)
# This allows better fine-tuning
for layer in base_model.layers[:-100]:
    layer.trainable = False

print(f"\nArchitecture:")
print(f"  Base: MobileNetV3Large")
print(f"  Frozen layers: {len([l for l in base_model.layers[:-100] if not l.trainable])}")
print(f"  Trainable layers: {len([l for l in base_model.layers[-100:] if l.trainable])}")

# Build full model with better head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    # First dense layer - strong but not excessive
    layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-5)),
    layers.ReLU(),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    # Second dense layer
    layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-5)),
    layers.ReLU(),
    layers.Dropout(0.25),
    layers.BatchNormalization(),
    # Third dense layer
    layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-5)),
    layers.ReLU(),
    layers.Dropout(0.2),
    # Output
    layers.Dense(len(class_names), activation='softmax')
])

print(f"  Total parameters: {model.count_params():,}")

# ============================================================================
# STEP 4: COMPILE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: COMPILING MODEL")
print("=" * 80)

# Higher learning rate for better learning
optimizer = keras.optimizers.Adam(learning_rate=5e-4)  # 10x higher than before!
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n✓ Model compiled:")
print(f"  Optimizer: Adam")
print(f"  Learning rate: 5e-4 (10x higher!)")
print(f"  Loss: Sparse Categorical Crossentropy")

# ============================================================================
# STEP 5: DATA AUGMENTATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DATA AUGMENTATION")
print("=" * 80)

# Training augmentation - aggressive but reasonable
train_datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True
)

# Validation augmentation - minimal
val_datagen = ImageDataGenerator()

print(f"\n✓ Augmentation configured:")
print(f"  Training: Rotation ±30°, Zoom ±30%, Brightness ±30%")
print(f"  Validation: No augmentation (for fair monitoring)")

# ============================================================================
# STEP 6: TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: TRAINING")
print("=" * 80)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # More patient than before
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5
    ),
    keras.callbacks.ModelCheckpoint(
        'media/pilldata/model_enhanced_best.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]

print(f"\nTraining configuration:")
print(f"  Epochs: 100 (with early stopping)")
print(f"  Batch size: 32")
print(f"  Early stopping patience: 20")
print(f"  Learning rate schedule: Yes")
print(f"\nStarting training...\n")

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=100,
    validation_data=val_datagen.flow(X_val, y_val, batch_size=32),
    validation_steps=len(X_val) // 32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: SAVING MODEL")
print("=" * 80)

model.save('media/pilldata/model_enhanced.keras')

# Save metadata
metadata = {
    'class_names': class_names,
    'image_size': [224, 224],
    'training_images': len(X_train),
    'validation_images': len(X_val),
    'total_params': model.count_params(),
    'training_epochs': len(history.history['loss']),
    'final_train_accuracy': float(history.history['accuracy'][-1]),
    'final_val_accuracy': float(history.history['val_accuracy'][-1]),
    'architecture': 'MobileNetV3Large with 3-layer head'
}

with open('media/pilldata/model_enhanced_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Model saved: media/pilldata/model_enhanced.keras")
print(f"✓ Metadata saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(f"\n✓ Model statistics:")
print(f"  Classes: {len(class_names)}")
print(f"  Training images: {len(X_train)}")
print(f"  Validation images: {len(X_val)}")
print(f"  Total parameters: {model.count_params():,}")
print(f"  Final training accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")

print(f"\n✓ Key improvements in this version:")
print(f"  1. Learning rate: 5e-5 → 5e-4 (10x higher)")
print(f"  2. Dropout: Up to 50% → 30% (lighter)")
print(f"  3. Regularization: Moderate (not too strong)")
print(f"  4. Trainable layers: 50 → 100 (more flexibility)")
print(f"  5. Early stopping patience: 30 → 20 (faster convergence)")

print(f"\n✓ Next steps:")
print(f"  1. Run: python compare_models.py")
print(f"  2. Compare old vs new model")
print(f"  3. Evaluate improvements")
print(f"  4. Deploy if validated")

print("\n" + "=" * 80)
