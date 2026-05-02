#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast Balanced Training - Optimized for Speed
Uses efficient loading and minimal augmentation for quick training
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Fix encoding on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import json


def get_data_paths():
    """Get portable data paths"""
    from django.conf import settings
    BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    return {
        'train_csv': BASE_DATA_PATH / 'Training_set.csv',
        'test_csv': BASE_DATA_PATH / 'Testing_set.csv',
        'train_path': BASE_DATA_PATH / 'train',
        'model_path': BASE_DATA_PATH / 'model_imprint_robust_v2.h5',
        'model_metadata': BASE_DATA_PATH / 'model_imprint_robust_v2_metadata.json',
    }


def load_images_efficiently(df, train_path, label_map, max_images=None):
    """Load images efficiently with batch processing"""
    images = []
    labels = []
    failed = 0
    
    total = min(len(df), max_images) if max_images else len(df)
    
    for idx, row in df.iterrows():
        if max_images and idx >= max_images:
            break
            
        img_path = Path(train_path) / row['filename']
        
        try:
            # Use PIL for fast loading
            img = load_img(str(img_path), target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            
            images.append(img_array)
            labels.append(label_map[row['label']])
            
            if (idx + 1) % 100 == 0:
                logger.info(f"  Loaded {idx + 1}/{total} images")
                
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"Failed to load {img_path}: {str(e)[:50]}")
    
    logger.info(f"Successfully loaded {len(images)} images ({failed} failed)")
    return np.array(images), np.array(labels)


def build_model(num_classes):
    """Build MobileNetV3 model"""
    logger.info("Building MobileNetV3 model...")
    
    base_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built with {num_classes} classes")
    return model


def train_model(model, train_images, train_labels, test_images, test_labels, model_path, epochs=50):
    """Train model with balanced augmentation"""
    
    logger.info("="*60)
    logger.info("TRAINING WITH BALANCED AUGMENTATION")
    logger.info("="*60)
    
    # Create augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False,
    )
    
    # Class weights
    num_classes = len(np.unique(train_labels))
    class_weights_dict = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_labels),
        train_labels
    )
    class_weights = {i: class_weights_dict[i] for i in range(num_classes)}
    
    logger.info("Class Weights:")
    for class_idx, weight in class_weights.items():
        logger.info(f"  Class {class_idx}: {weight:.3f}")
    
    # Convert labels
    train_labels_cat = to_categorical(train_labels, num_classes=num_classes)
    test_labels_cat = to_categorical(test_labels, num_classes=num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(model_path), monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train
    batch_size = 32
    steps_per_epoch = max(1, len(train_images) // batch_size)
    
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Total images: {len(train_images)}")
    logger.info("\nStarting training...\n")
    
    history = model.fit(
        train_datagen.flow(train_images, train_labels_cat, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(test_images, test_labels_cat),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_images, test_labels, reverse_label_map):
    """Evaluate model on test set"""
    logger.info("\n" + "="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    test_labels_cat = to_categorical(test_labels, num_classes=len(reverse_label_map))
    
    # Overall evaluation
    loss, accuracy = model.evaluate(test_images, test_labels_cat, verbose=0)
    logger.info(f"\nOverall Accuracy: {accuracy:.2%}")
    logger.info(f"Loss: {loss:.4f}")
    
    # Per-class evaluation
    predictions = model.predict(test_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    logger.info("\nPer-Class Accuracy:")
    for class_idx in sorted(reverse_label_map.keys()):
        mask = test_labels == class_idx
        if mask.sum() > 0:
            class_accuracy = (predicted_labels[mask] == class_idx).mean()
            logger.info(f"  {reverse_label_map[class_idx]}: {class_accuracy:.2%} ({mask.sum()} samples)")


def main():
    """Main execution"""
    paths = get_data_paths()
    
    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(paths['train_csv'])
    test_df = pd.read_csv(paths['test_csv'])
    
    # Create label mapping
    unique_labels = sorted(set(train_df['label'].unique()) | set(test_df['label'].unique()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    
    logger.info(f"Found {len(unique_labels)} pill classes")
    logger.info(f"Classes: {unique_labels}")
    
    # Load training images
    logger.info("\nLoading training images...")
    train_images, train_labels = load_images_efficiently(
        train_df, paths['train_path'], label_map
    )
    
    # Load test images
    logger.info("\nLoading test images...")
    test_images, test_labels = load_images_efficiently(
        test_df, paths['train_path'], label_map
    )
    
    # Build model
    model = build_model(len(unique_labels))
    
    # Train model
    history = train_model(
        model,
        train_images,
        train_labels,
        test_images,
        test_labels,
        paths['model_path'],
        epochs=50
    )
    
    # Evaluate
    evaluate_model(model, test_images, test_labels, reverse_label_map)
    
    # Save metadata
    logger.info("\nSaving model metadata...")
    metadata = {
        'classes': unique_labels,
        'label_map': {str(k): int(v) for k, v in label_map.items()},
        'reverse_label_map': {int(k): str(v) for k, v in reverse_label_map.items()},
        'confidence_threshold': 0.50,
        'model_type': 'MobileNetV3Large',
        'training_approach': 'balanced_augmentation_v2'
    }
    
    with open(paths['model_metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Model saved: {paths['model_path']}")
    logger.info(f"✓ Metadata saved: {paths['model_metadata']}")
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
