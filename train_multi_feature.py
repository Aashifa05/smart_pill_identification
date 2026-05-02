#!/usr/bin/env python3
"""
Multi-Feature Enhanced Training
================================
Train model using shape, color, size, texture, and imprint features
Improves discrimination between known and unknown pills
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("MULTI-FEATURE ENHANCED TRAINING")
print("=" * 80)
print(f"\nTraining strategy:")
print("  • Extract: Shape, Color, Size, Texture, Imprint features")
print("  • Combine: Features + Deep Learning hybrid approach")
print("  • Focus: Better discrimination for known vs unknown pills")
print("  • Goal: Minimize false negatives (known marked unknown)")
print("         Minimize false positives (unknown marked as known)")

# ============================================================================
# STEP 1: Load and Organize Data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING TRAINING DATA")
print("=" * 80)

train_dir = Path('media/pilldata/train')
test_dir = Path('media/pilldata/test')

# Get class mapping from metadata
metadata_path = Path('media/pilldata/model_metadata.json')
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        old_metadata = json.load(f)
    class_names = list(old_metadata.get('label_map', {}).keys())
else:
    class_names = []

print(f"Known classes ({len(class_names)}):")
for i, name in enumerate(class_names, 1):
    print(f"  {i:2}. {name}")

# Organize training images by class
class_images = defaultdict(list)
for img_path in sorted(train_dir.glob('*.jpg')) + sorted(train_dir.glob('*.png')):
    # Extract class from filename
    filename = img_path.stem
    # Remove copy markers
    clean_name = filename.replace(' - Copy', '').replace(' - Copy', '').strip()
    # Remove trailing number if exists
    parts = clean_name.rsplit(' ', 1)
    if parts[-1].isdigit():
        clean_name = ' '.join(parts[:-1])
    
    class_images[clean_name].append(img_path)

print(f"\nFound images for {len(class_images)} classes")
print(f"Total images: {sum(len(v) for v in class_images.values())}")

# ============================================================================
# STEP 2: Load Feature Extractor
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: LOADING FEATURE EXTRACTORS")
print("=" * 80)

try:
    from Users.utility.multi_feature_pill_classifier import MultiFeaturePillClassifier
    
    feature_extractor = MultiFeaturePillClassifier()
    print("✓ Feature extractor loaded")
    print("  • Shape features (5 metrics)")
    print("  • Color features (7 metrics)")
    print("  • Size features (6 metrics)")
    print("  • Texture features (6 metrics)")
    print("  • Imprint features (7 metrics)")
    print("  Total: 31 handcrafted features")
except Exception as e:
    print(f"⚠️  Could not load feature extractor: {e}")
    print("  Will rely on deep learning only")
    feature_extractor = None

# ============================================================================
# STEP 3: Prepare Training Data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: PREPARING TRAINING DATA WITH FEATURES")
print("=" * 80)

from tensorflow.keras.preprocessing import image

X_train = []
X_features = []
y_train = []
class_to_idx = {name: i for i, name in enumerate(class_names)}

print(f"Processing images for {len(class_names)} known classes...")

for class_idx, class_name in enumerate(class_names):
    # Get images for this class
    images = []
    for key, paths in class_images.items():
        if key.lower() == class_name.lower():
            images.extend(paths)
    
    if not images:
        print(f"  ⚠️  No images found for {class_name}")
        continue
    
    print(f"  [{class_idx+1:2}/{len(class_names)}] {class_name:40} ({len(images)} images)")
    
    for img_path in images:
        try:
            # Load image for CNN
            img = image.load_img(str(img_path), target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            X_train.append(img_array)
            
            # Extract features if possible
            if feature_extractor:
                try:
                    features = feature_extractor.extract_features(str(img_path))
                    # Convert features dict to vector
                    feature_vector = []
                    for key in sorted(features.keys()):
                        val = features[key]
                        if isinstance(val, (int, float)):
                            feature_vector.append(val)
                        elif isinstance(val, bool):
                            feature_vector.append(1.0 if val else 0.0)
                        elif isinstance(val, (list, tuple)):
                            feature_vector.extend(val)
                    X_features.append(np.array(feature_vector))
                except:
                    X_features.append(np.zeros(31))  # Default feature vector
            
            y_train.append(class_to_idx[class_name])
        except Exception as e:
            print(f"    Error processing {img_path.name}: {e}")

X_train = np.array(X_train)
y_train = np.array(y_train)
X_features = np.array(X_features) if X_features else None

print(f"\n✓ Data prepared:")
print(f"  Images: {X_train.shape}")
if X_features is not None:
    print(f"  Features: {X_features.shape}")
print(f"  Classes: {len(class_names)}")

# ============================================================================
# STEP 4: Build Enhanced Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: BUILDING MULTI-FEATURE ENHANCED MODEL")
print("=" * 80)

from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
    Concatenate, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2

# Image input path
image_input = Input(shape=(224, 224, 3), name='image_input')

# CNN branch: MobileNetV3
base_model = MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
# Freeze early layers, train later ones
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model(image_input)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu', kernel_regularizer=L1L2(1e-4, 1e-4))(x)
x = Dropout(0.45)(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=L1L2(1e-4, 1e-4))(x)
x = Dropout(0.35)(x)
x = BatchNormalization()(x)
cnn_output = Dense(256, activation='relu')(x)

# Feature input path (if available)
if X_features is not None and X_features.shape[1] > 0:
    feature_input = Input(shape=(X_features.shape[1],), name='feature_input')
    f = Dense(128, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5))(feature_input)
    f = Dropout(0.3)(f)
    f = Dense(64, activation='relu')(f)
    f = Dropout(0.2)(f)
    feature_output = Dense(64, activation='relu')(f)
    
    # Combine paths
    combined = Concatenate()([cnn_output, feature_output])
else:
    feature_input = None
    combined = cnn_output

# Final classification layers
combined = Dense(256, activation='relu', kernel_regularizer=L1L2(1e-4, 1e-4))(combined)
combined = Dropout(0.4)(combined)
combined = BatchNormalization()(combined)
combined = Dense(128, activation='relu')(combined)
combined = Dropout(0.3)(combined)
output = Dense(len(class_names), activation='softmax', name='predictions')(combined)

# Build model
if feature_input is not None:
    model = Model(inputs=[image_input, feature_input], outputs=output)
    print("✓ Built hybrid model (CNN + Features)")
else:
    model = Model(inputs=image_input, outputs=output)
    print("✓ Built CNN-only model (no handcrafted features)")

print(f"  Parameters: {model.count_params():,}")
model.summary(expand_nested=True)

# ============================================================================
# STEP 5: Compile Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: COMPILING MODEL")
print("=" * 80)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled")
print("  Optimizer: Adam (lr=1e-4)")
print("  Loss: sparse_categorical_crossentropy")
print("  Focus: Maximize accuracy on training data")

# ============================================================================
# STEP 6: Train Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: TRAINING MODEL")
print("=" * 80)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Strong augmentation for better generalization
train_augmentation = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation augmentation (lighter)
val_augmentation = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Split into train/val
from sklearn.model_selection import train_test_split

if feature_input is not None:
    X_train_split, X_val_split, F_train_split, F_val_split, y_train_split, y_val_split = train_test_split(
        X_train, X_features, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create data generators that work with both inputs
    def create_data_generator(images, features, labels, augmentation, batch_size=32):
        while True:
            indices = np.random.choice(len(images), batch_size)
            batch_images = images[indices]
            batch_features = features[indices]
            batch_labels = labels[indices]
            
            # Apply augmentation
            for i in range(len(batch_images)):
                batch_images[i] = augmentation.random_transform(batch_images[i])
            
            yield [batch_images, batch_features], batch_labels
    
    train_gen = create_data_generator(X_train_split, F_train_split, y_train_split, train_augmentation)
    val_gen = create_data_generator(X_val_split, F_val_split, y_val_split, val_augmentation)
    
    steps_per_epoch = len(X_train_split) // 32
    validation_steps = len(X_val_split) // 32
else:
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    train_gen = train_augmentation.flow(
        X_train_split, y_train_split, batch_size=32, shuffle=True
    )
    val_gen = val_augmentation.flow(
        X_val_split, y_val_split, batch_size=32, shuffle=False
    )
    
    steps_per_epoch = len(X_train_split) // 32
    validation_steps = len(X_val_split) // 32

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"Training configuration:")
print(f"  Train samples: {len(X_train_split)}")
print(f"  Validation samples: {len(X_val_split)}")
print(f"  Batch size: 32")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Max epochs: 150")
print(f"  Early stopping patience: 25 epochs")

print(f"\nStarting training...")
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=150,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# STEP 7: Evaluate on Test Set
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: EVALUATING ON TEST DATA")
print("=" * 80)

# Load test images
test_images = []
test_labels = []

for img_path in sorted(test_dir.glob('*.jpg')) + sorted(test_dir.glob('*.png')):
    # Extract class from filename
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
        except:
            pass

if test_images:
    X_test = np.array(test_images)
    y_test = np.array(test_labels)
    
    if feature_input is not None:
        # Extract features for test set
        test_features = []
        for img_path in sorted(test_dir.glob('*.jpg')) + sorted(test_dir.glob('*.png')):
            if feature_extractor:
                try:
                    features = feature_extractor.extract_features(str(img_path))
                    feature_vector = []
                    for key in sorted(features.keys()):
                        val = features[key]
                        if isinstance(val, (int, float)):
                            feature_vector.append(val)
                        elif isinstance(val, bool):
                            feature_vector.append(1.0 if val else 0.0)
                        elif isinstance(val, (list, tuple)):
                            feature_vector.extend(val)
                    test_features.append(np.array(feature_vector))
                except:
                    test_features.append(np.zeros(31))
        
        X_test_features = np.array(test_features)
        test_loss, test_acc = model.evaluate([X_test, X_test_features], y_test, verbose=0)
    else:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2%}")
else:
    print("No test images found")
    test_acc = 0

# ============================================================================
# STEP 8: Save Model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: SAVING MODEL")
print("=" * 80)

model_path = Path('media/pilldata/model_multi_feature.keras')
model_path.parent.mkdir(parents=True, exist_ok=True)

model.save(str(model_path))
print(f"✓ Model saved: {model_path}")

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'type': 'multi_feature_enhanced',
    'input_shape': [224, 224, 3],
    'num_classes': len(class_names),
    'label_map': class_to_idx,
    'class_names': class_names,
    'training_samples': len(X_train),
    'test_accuracy': float(test_acc),
    'architecture': 'MobileNetV3Large + Features + Dense layers',
    'features_used': ['shape', 'color', 'size', 'texture', 'imprint'],
    'augmentation': 'strong (rotation, zoom, brightness, shifts)',
    'regularization': 'L1L2 + Dropout + BatchNormalization',
    'callbacks': ['EarlyStopping(patience=25)', 'ReduceLROnPlateau'],
    'training_notes': 'Multi-feature enhanced model trained to better discriminate known vs unknown pills'
}

metadata_path = Path('media/pilldata/model_multi_feature_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved: {metadata_path}")

# ============================================================================
# STEP 9: Summary
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\n✓ Model: {model_path}")
print(f"✓ Classes: {len(class_names)}")
print(f"✓ Training accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"✓ Validation accuracy: {history.history['val_accuracy'][-1]:.2%}")
print(f"✓ Test accuracy: {test_acc:.2%}")

print(f"\nModel improvements:")
print(f"  • Multi-feature extraction (31 handcrafted features)")
print(f"  • Hybrid CNN + feature architecture")
print(f"  • Strong regularization to prevent overfitting")
print(f"  • Transfer learning from ImageNet")
print(f"  • Aggressive data augmentation")
print(f"  • Early stopping to find optimal weights")

print(f"\nNext steps:")
print(f"  1. Use this model: model_multi_feature.keras")
print(f"  2. Test on known pill images")
print(f"  3. Test on unknown pill images")
print(f"  4. Adjust confidence thresholds if needed")
print(f"  5. Deploy to production")

print("\n" + "=" * 80)
