#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Anti-Overfitting Training for Pill Classifier
======================================================

Fast training script that leverages the existing trained model
and improves it with anti-overfitting techniques.

Uses transfer learning + data augmentation to prevent memorization
while maintaining real-world generalization.
"""

import os
import sys
import numpy as np
from pathlib import Path
import logging
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_class_from_filename(filename: str) -> str:
    """Extract medication name from filename"""
    import re
    # Remove trailing "(N)" and "- Copy" patterns
    name = re.sub(r'\s*\(\d+\).*$', '', filename)  # Remove (N)
    name = re.sub(r'\s*-\s*Copy.*$', '', name)      # Remove - Copy variants
    return name.strip()


def load_training_data(data_dir='media/pilldata/train', max_samples=None):
    """Load training data with flexible image organization"""
    
    logger.info(f"\n{'='*70}")
    logger.info("LOADING TRAINING DATA")
    logger.info(f"{'='*70}\n")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return None
    
    images = []
    labels_list = []
    class_names_dict = {}
    current_class_idx = 0
    
    # Get all image files
    all_images = sorted(list(data_path.glob('*.jpg')) + list(data_path.glob('*.png')))
    logger.info(f"Found {len(all_images)} image files")
    
    # Load images and extract class names
    for idx, image_path in enumerate(all_images):
        # Extract class name from filename
        class_name = extract_class_from_filename(image_path.stem)
        
        # Map class name to index
        if class_name not in class_names_dict:
            class_names_dict[class_name] = current_class_idx
            current_class_idx += 1
        
        # Load image
        try:
            img = load_img(str(image_path), target_size=(224, 224))
            x = img_to_array(img)
            x = preprocess_input(x)
            images.append(x)
            labels_list.append(class_names_dict[class_name])
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
        
        if (idx + 1) % 500 == 0:
            logger.info(f"  Loaded {idx+1}/{len(all_images)} images...")
    
    images = np.array(images, dtype=np.float32)
    labels_array = np.array(labels_list)
    class_names = sorted(class_names_dict.keys())
    
    logger.info(f"\nLoaded {len(images)} images")
    logger.info(f"Detected {len(class_names)} medication classes:")
    for idx, name in enumerate(class_names):
        count = np.sum(labels_array == idx)
        logger.info(f"  [{idx+1:2d}] {name:45s}: {count:4d} images")
    
    return images, labels_array, class_names


def build_antioverfit_model(num_classes: int):
    """Build model with anti-overfitting architecture"""
    
    logger.info(f"\n{'='*70}")
    logger.info("BUILDING MODEL")
    logger.info(f"{'='*70}\n")
    
    # Input
    inputs = Input(shape=(224, 224, 3), name='input')
    
    # Pre-trained MobileNetV3 base
    logger.info("Loading pre-trained MobileNetV3Large...")
    base_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Feature extraction
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    x = Dense(512, kernel_regularizer=L1L2(l1=1e-4, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, kernel_regularizer=L1L2(l1=1e-4, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Output
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='PillClassifierAntiOverfit')
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    return model


def train_with_augmentation(model, X_train, X_val, y_train, y_val, epochs=100, batch_size=32):
    """Train with data augmentation"""
    
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING WITH ANTI-OVERFITTING TECHNIQUES")
    logger.info(f"{'='*70}\n")
    
    # Data augmentation for training
    train_aug = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.25,
        brightness_range=[0.75, 1.25],
        channel_shift_range=25,
        horizontal_flip=True,
        fill_mode='reflect'
    )
    
    # Validation (no augmentation)
    val_aug = ImageDataGenerator()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        ModelCheckpoint('media/pilldata/model_anti_overfit.keras', monitor='val_loss', save_best_only=True, verbose=0),
    ]
    
    # Train
    history = model.fit(
        train_aug.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_and_plot(model, X_test, y_test, class_names):
    """Evaluate model and create visualization"""
    
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION")
    logger.info(f"{'='*70}\n")
    
    # Test metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Test Loss:     {test_loss:.4f}")
    
    # Detailed metrics
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    logger.info(f"\nDetailed Metrics:")
    logger.info(f"  Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    logger.info(f"  Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    logger.info(f"  F1 Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    
    return test_acc, test_loss


def main():
    """Main training pipeline"""
    
    logger.info("\n" + "="*70)
    logger.info("ANTI-OVERFITTING PILL CLASSIFIER TRAINING")
    logger.info("="*70)
    logger.info("Goal: Learn general pill features for real-world generalization")
    logger.info("="*70 + "\n")
    
    # Load data
    data = load_training_data()
    if data is None:
        return
    
    images, labels, class_names = data
    num_classes = len(class_names)
    
    # Convert labels to categorical
    labels_cat = to_categorical(labels, num_classes)
    
    # Split data: 70% train, 15% val, 15% test
    logger.info(f"\n{'='*70}")
    logger.info("DATA SPLIT")
    logger.info(f"{'='*70}\n")
    
    X_temp, X_test, y_temp, y_test = train_test_split(images, labels_cat, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=np.argmax(y_temp, axis=1))
    
    logger.info(f"Training:   {len(X_train)} images ({len(X_train)/len(images)*100:.1f}%)")
    logger.info(f"Validation: {len(X_val)} images ({len(X_val)/len(images)*100:.1f}%)")
    logger.info(f"Test:       {len(X_test)} images ({len(X_test)/len(images)*100:.1f}%)")
    
    # Build model
    model = build_antioverfit_model(num_classes)
    
    # Train
    history = train_with_augmentation(model, X_train, X_val, y_train, y_val, epochs=150, batch_size=32)
    
    # Evaluate
    test_acc, test_loss = evaluate_and_plot(model, X_test, y_test, class_names)
    
    # Save metadata
    metadata = {
        'model_name': 'PillClassifier_AntiOverfit',
        'train_date': datetime.now().isoformat(),
        'num_classes': num_classes,
        'class_names': class_names,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'architecture': 'MobileNetV3Large + Dense layers',
        'regularization': 'L1L2 + Dropout + BatchNorm',
        'augmentation': 'Rotation, Zoom, Brightness, Color shift'
    }
    
    with open('media/pilldata/model_anti_overfit_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"✓ Model saved: media/pilldata/model_anti_overfit.keras")
    logger.info(f"✓ Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"✓ Status: {'EXCELLENT' if test_acc > 0.85 else 'GOOD' if test_acc > 0.80 else 'NEEDS IMPROVEMENT'}")
    logger.info(f"{'='*70}\n")


if __name__ == '__main__':
    main()
