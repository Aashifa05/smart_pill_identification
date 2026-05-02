#!/usr/bin/env python
"""MINIMAL training - just 3 epochs to get model structure"""

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
from tensorflow.keras.applications import MobileNetV3Small  # Smaller model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_sample_images(max_per_class=50):
    """Load limited sample of images for quick training"""
    logger.info(f"Loading sample images (max {max_per_class} per class)...")
    dataset_path = Path("media/pilldata/train")
    
    images = []
    labels = []
    label_map = {}
    class_idx = 0
    
    def extract_pill_name(filename):
        """Extract pill name from filename"""
        match = re.match(r'([^0-9]*?)\s+\d+', filename)
        if match:
            return match.group(1).strip()
        
        match = re.match(r'([a-zA-Z0-9]+-?[a-zA-Z0-9]*?)_', filename)
        if match:
            return match.group(1).strip()
        
        match = re.match(r'([a-zA-Z0-9]+-[a-zA-Z0-9]+)', filename)
        if match:
            return match.group(1).strip()
        
        return None
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_img_files = []
    for ext in image_extensions:
        all_img_files.extend(list(dataset_path.glob(ext)))
    
    # Group by pill class
    pill_groups = {}
    for img_path in all_img_files:
        pill_name = extract_pill_name(img_path.stem)
        if not pill_name:
            continue
        if pill_name not in pill_groups:
            pill_groups[pill_name] = []
        pill_groups[pill_name].append(img_path)
    
    # Load limited samples per class
    for pill_name in sorted(pill_groups.keys()):
        label_map[pill_name] = class_idx
        class_idx += 1
        
        # Take only first N images per class
        for img_path in pill_groups[pill_name][:max_per_class]:
            try:
                img = load_img(str(img_path), target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label_map[pill_name])
            except Exception as e:
                logger.warning(f"Skipped: {e}")
    
    logger.info(f"✓ Loaded {len(images)} sample images from {len(label_map)} classes")
    return np.array(images), np.array(labels), label_map

def build_model(num_classes):
    """Build minimal model for speed"""
    logger.info("Building MobileNetV3Small model...")
    
    inputs = Input(shape=(224, 224, 3))
    
    # Use smaller model
    mobilenet = MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze all backbone layers
    for layer in mobilenet.layers:
        layer.trainable = False
    
    # Minimal head
    x = mobilenet(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train():
    """Minimal training"""
    logger.info("=" * 80)
    logger.info("MINIMAL TRAINING: Quick Model Structure Test")
    logger.info("=" * 80 + "\n")
    
    # Load sample data
    X, y, label_map = load_sample_images(max_per_class=30)
    y = to_categorical(y, num_classes=len(label_map))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")
    
    # Build & compile
    model = build_model(len(label_map))
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for just 3 epochs
    logger.info("Training for 3 epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=16,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"\n✓ Test Accuracy: {test_acc:.4f}")
    
    # Save
    model.save('media/pilldata/model_feature_learning.keras')
    logger.info("✓ Model saved")
    
    # Save metadata
    metadata = {
        'class_names': sorted(label_map.items(), key=lambda x: x[1]),
        'num_classes': len(label_map),
        'training_samples': len(X_train),
        'model_type': 'MobileNetV3Small',
        'test_accuracy': float(test_acc),
        'note': 'Quick training model - use for testing only'
    }
    
    with open('media/pilldata/model_feature_learning_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✓ Metadata saved\n")
    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
