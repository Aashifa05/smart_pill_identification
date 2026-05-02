#!/usr/bin/env python
"""Quick pill classifier training - no augmentation for speed"""

import os
import sys
import re
import logging
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_images():
    """Load all pill images from flat directory"""
    logger.info("Loading images...")
    dataset_path = Path("media/pilldata/train")
    
    images = []
    labels = []
    label_map = {}
    class_idx = 0
    
    def extract_pill_name(filename):
        """Extract pill name from filename"""
        # Pattern 1: PillName dosage MG (number)
        match = re.match(r'([^0-9]*?)\s+\d+', filename)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: NDC-Code_index
        match = re.match(r'([a-zA-Z0-9]+-?[a-zA-Z0-9]*?)_', filename)
        if match:
            return match.group(1).strip()
        
        # Pattern 3: Part before dash
        match = re.match(r'([a-zA-Z0-9]+-[a-zA-Z0-9]+)', filename)
        if match:
            return match.group(1).strip()
        
        return None
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_img_files = []
    for ext in image_extensions:
        all_img_files.extend(list(dataset_path.glob(ext)))
    
    # Build label map and load images
    pill_classes = {}
    for img_path in sorted(all_img_files):
        pill_name = extract_pill_name(img_path.stem)
        if not pill_name:
            continue
        
        if pill_name not in label_map:
            label_map[pill_name] = class_idx
            class_idx += 1
            pill_classes[pill_name] = 0
        
        pill_classes[pill_name] += 1
        
        try:
            img = load_img(str(img_path), target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label_map[pill_name])
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
    
    logger.info(f"✓ Loaded {len(images)} images from {len(label_map)} classes")
    for pill_name in sorted(pill_classes.keys()):
        logger.info(f"  {pill_name}: {pill_classes[pill_name]} images")
    
    return np.array(images), np.array(labels), label_map

def build_model(num_classes):
    """Build simple MobileNetV3 classifier"""
    logger.info("Building model...")
    
    inputs = Input(shape=(224, 224, 3))
    
    # Load MobileNetV3 backbone
    mobilenet = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze most layers
    for layer in mobilenet.layers[:-15]:
        layer.trainable = False
    
    # Feature extraction
    x = mobilenet(inputs)
    x = GlobalAveragePooling2D()(x)
    
    # Dense head
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Classification
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model():
    """Train the model"""
    logger.info("=" * 80)
    logger.info("QUICK TRAINING: Pill Classifier (No Augmentation)")
    logger.info("=" * 80 + "\n")
    
    # Load data
    X, y, label_map = load_images()
    
    # Convert labels
    y = to_categorical(y, num_classes=len(label_map))
    
    # Split data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")
    
    # Build model
    model = build_model(len(label_map))
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(model.summary())
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'media/pilldata/model_feature_learning.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Train
    logger.info("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,  # Quick training
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Save metadata
    metadata = {
        'class_names': sorted(label_map.items(), key=lambda x: x[1]),
        'num_classes': len(label_map),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': float(test_acc),
        'model_type': 'MobileNetV3Large',
        'input_shape': [224, 224, 3]
    }
    
    with open('media/pilldata/model_feature_learning_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Model saved: media/pilldata/model_feature_learning.keras")
    logger.info(f"✅ Metadata saved: media/pilldata/model_feature_learning_metadata.json")
    logger.info("=" * 80)
    
    return model, history

if __name__ == '__main__':
    try:
        model, history = train_model()
        logger.info("\n✅ TRAINING COMPLETE!")
        logger.info("\nNext: python run_model_diagnostic.py --model model_feature_learning.keras")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
