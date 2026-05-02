#!/usr/bin/env python
"""
Simple and Fast Training for Pharmaceutical Pill Classification
Optimized for speed - no strict validation
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
MEDIA_DIR = BASE_DIR / 'media' / 'pilldata'
TRAIN_IMG_DIR = MEDIA_DIR / 'train'
TEST_IMG_DIR = MEDIA_DIR / 'test'
TRAIN_CSV = MEDIA_DIR / 'Training_set.csv'
TEST_CSV = MEDIA_DIR / 'Testing_set.csv'
MODEL_PATH = MEDIA_DIR / 'model.keras'

# Model parameters
INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

def load_images(image_paths, labels, img_dir):
    """Load and preprocess images"""
    images = []
    valid_labels = []
    
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        try:
            full_path = img_dir / img_path
            if not full_path.exists():
                logger.warning(f"  ⚠️  Missing: {img_path}")
                continue
            
            img = load_img(str(full_path), target_size=(INPUT_SIZE, INPUT_SIZE))
            img_array = img_to_array(img) / 255.0
            
            images.append(img_array)
            valid_labels.append(label)
            
            if (idx + 1) % 200 == 0:
                logger.info(f"  ✓ Loaded {idx + 1}/{len(image_paths)} images...")
                
        except Exception as e:
            logger.warning(f"  ⚠️  Error with {img_path}: {e}")
            continue
    
    logger.info(f"  ✓ Successfully loaded {len(images)} images")
    return np.array(images, dtype=np.float32), np.array(valid_labels)

def train():
    """Main training function"""
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING MODEL TRAINING - Pharmaceutical Pill Classification")
    logger.info("=" * 80)
    
    # Load CSVs
    logger.info("\n📂 Loading dataset metadata...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logger.info(f"  Training: {len(train_df)} samples")
    logger.info(f"  Testing: {len(test_df)} samples")
    
    # Create label mapping
    logger.info("\n🏷️  Creating label mapping...")
    unique_labels = sorted(train_df['label'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    num_classes = len(label_map)
    logger.info(f"  Classes: {num_classes}")
    for i in range(min(5, num_classes)):
        logger.info(f"    {i}: {reverse_label_map[i]}")
    if num_classes > 5:
        logger.info(f"    ... and {num_classes - 5} more classes")
    
    # Convert to indices
    train_labels = train_df['label'].map(label_map).values
    test_labels = test_df['label'].map(label_map).values
    
    # Load images
    logger.info("\n🖼️  Loading training images...")
    X_train, y_train = load_images(train_df['filename'].values, train_labels, TRAIN_IMG_DIR)
    
    logger.info("\n🖼️  Loading test images...")
    X_test, y_test = load_images(test_df['filename'].values, test_labels, TEST_IMG_DIR)
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("❌ No images loaded!")
        return None, None, None, None
    
    # One-hot encode
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Build model
    logger.info("\n🏗️  Building MobileNetV3 model...")
    base_model = MobileNetV3Large(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    logger.info("  ✓ Base model loaded (ImageNet weights)")
    
    inputs = base_model.input
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    logger.info("  ✓ Model compiled")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    # Train
    logger.info(f"\n🎯 Training for {EPOCHS} epochs...")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_test)}")
    
    history = model.fit(
        X_train, y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    logger.info("\n📊 Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"  Test Loss: {loss:.4f}")
    logger.info(f"  Test Accuracy: {accuracy:.2%}")
    
    # Save model
    logger.info(f"\n💾 Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    logger.info("  ✓ Model saved")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_classes': num_classes,
        'input_shape': [INPUT_SIZE, INPUT_SIZE, 3],
        'label_map': {str(k): int(v) for k, v in label_map.items()},
        'reverse_label_map': {int(k): str(v) for k, v in reverse_label_map.items()},
        'accuracy': float(accuracy),
        'loss': float(loss),
        'epochs': len(history.history['loss']),
    }
    
    with open(MEDIA_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("  ✓ Metadata saved")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"📊 Final Accuracy: {accuracy:.2%}")
    logger.info(f"📊 Final Loss: {loss:.4f}")
    logger.info(f"💾 Model: {MODEL_PATH}")
    logger.info("=" * 80 + "\n")
    
    return accuracy, model, label_map, reverse_label_map

if __name__ == "__main__":
    train()
