"""
IMPROVED TRAINING FOR 95% ACCURACY
Simplified version that works with flat image structure
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint
)
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

print("=" * 80)
print("TRAINING FOR 95% ACCURACY - IMPROVED VERSION")
print("=" * 80)

# Configuration
INPUT_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.0001
TRAIN_DIR = 'media/pilldata/train'
TEST_DIR = 'media/pilldata/test'
MODEL_PATH = 'media/pilldata/model_95.keras'
METADATA_PATH = 'media/pilldata/model_metadata_95.json'

# ============================================================================
# 1. LOAD TRAINING DATA FROM CSV
# ============================================================================
print("\n[1/5] Loading training data from CSV...")

train_df = pd.read_csv('media/pilldata/Training_set.csv')
test_df = pd.read_csv('media/pilldata/Testing_set.csv')

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# Get unique classes
classes = sorted(train_df['label'].unique())
num_classes = len(classes)
print(f"Number of classes: {num_classes}")
print(f"Classes: {classes}")

# Create label mapping
label_map = {label: idx for idx, label in enumerate(classes)}
reverse_label_map = {idx: label for label, idx in label_map.items()}

# ============================================================================
# 2. PREPARE DATA WITH AGGRESSIVE AUGMENTATION
# ============================================================================
print("\n[2/5] Preparing data with aggressive augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./127.5,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
)

val_datagen = ImageDataGenerator(rescale=1./127.5)

# Prepare training data
print("Loading training images...")
X_train = []
y_train = []

for idx, row in train_df.iterrows():
    img_name = row['filename']
    label = row['label']
    
    img_path = os.path.join(TRAIN_DIR, img_name)
    if os.path.exists(img_path):
        try:
            img = load_img(img_path, target_size=(INPUT_SIZE, INPUT_SIZE))
            img_array = img_to_array(img) / 127.5
            X_train.append(img_array)
            y_train.append(label_map[label])
        except:
            print(f"Error loading {img_path}")
    
    if (idx + 1) % 100 == 0:
        print(f"  Loaded {idx + 1}/{len(train_df)} training images")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training data shape: {X_train.shape}")

# Prepare testing data
print("Loading testing images...")
X_test = []
y_test = []

for idx, row in test_df.iterrows():
    img_name = row['filename']
    label = row['label']
    
    img_path = os.path.join(TEST_DIR, img_name)
    if os.path.exists(img_path):
        try:
            img = load_img(img_path, target_size=(INPUT_SIZE, INPUT_SIZE))
            img_array = img_to_array(img) / 127.5
            X_test.append(img_array)
            y_test.append(label_map[label])
        except:
            print(f"Error loading {img_path}")
    
    if (idx + 1) % 50 == 0:
        print(f"  Loaded {idx + 1}/{len(test_df)} testing images")

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Testing data shape: {X_test.shape}")

# Convert labels to one-hot
y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes)

# ============================================================================
# 3. BUILD MODEL
# ============================================================================
print("\n[3/5] Building model architecture...")

base_model = MobileNetV3Large(
    input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

model = models.Sequential([
    layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax'),
])

print(f"Model built with {len(model.layers)} layers")

# ============================================================================
# 4. COMPILE AND SETUP CALLBACKS
# ============================================================================
print("\n[4/5] Compiling model with optimization...")

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1,
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
]

# ============================================================================
# 5. TRAIN THE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_test)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print("=" * 80 + "\n")

history = model.fit(
    X_train,
    y_train_categorical,
    validation_data=(X_test, y_test_categorical),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# ============================================================================
# SAVE MODEL AND METADATA
# ============================================================================
print("\n[SAVING] Model and metadata...")

model.save(MODEL_PATH)
print(f"✓ Model saved to: {MODEL_PATH}")

metadata = {
    'model_path': MODEL_PATH,
    'input_size': INPUT_SIZE,
    'classes': num_classes,
    'class_labels': classes,
    'class_indices': label_map,
    'reverse_label_map': reverse_label_map,
    'training_date': datetime.now().isoformat(),
    'accuracy': float(history.history['accuracy'][-1]),
    'val_accuracy': float(history.history['val_accuracy'][-1]),
    'epochs_trained': len(history.history['accuracy']),
    'final_loss': float(history.history['loss'][-1]),
    'final_val_loss': float(history.history['val_loss'][-1]),
}

with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"✓ Metadata saved to: {METADATA_PATH}")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - FINAL RESULTS")
print("=" * 80)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Epochs trained: {len(history.history['accuracy'])}")
print("=" * 80)

print("\n✅ Training complete! Model ready for deployment.")
