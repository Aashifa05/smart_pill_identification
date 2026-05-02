#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved Training with Balanced Augmentation
==============================================

This version uses LESS AGGRESSIVE augmentation to preserve class information
while still handling unlabeled pills. The key is balance:
- Keep enough detail for the model to learn classes
- Add some variations for robustness to imprints
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Fix encoding on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from PIL import Image, ImageFilter, ImageEnhance
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from datetime import datetime
import json


class ModerateAugmentation:
    """BALANCED augmentation - enough variation without losing class info"""
    
    @staticmethod
    def slight_blur(image, kernel_size=3):
        """Slight blur (NOT aggressive like before)"""
        img = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        result = np.array(blurred, dtype=np.float32) / 255.0 if image.dtype == np.float32 else np.array(blurred)
        return result
    
    @staticmethod
    def slight_contrast_adjustment(image, factor=0.85):
        """Slight contrast change (preserve details)"""
        img = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(img)
        adjusted = enhancer.enhance(factor)
        result = np.array(adjusted, dtype=np.float32) / 255.0 if image.dtype == np.float32 else np.array(adjusted)
        return result
    
    @staticmethod
    def slight_brightness(image, factor=1.1):
        """Slight brightness adjustment"""
        img = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(img)
        adjusted = enhancer.enhance(factor)
        result = np.array(adjusted, dtype=np.float32) / 255.0 if image.dtype == np.float32 else np.array(adjusted)
        return result
    
    @staticmethod
    def light_noise(image, noise_level=0.02):
        """Light noise (NOT aggressive)"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = np.clip(image + noise, 0, 1 if image.dtype == np.float32 else 255)
        return noisy


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


def load_and_preprocess_image(file_path, target_size=(224, 224)):
    """Load and preprocess single image"""
    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype='float32')
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def load_data(train_csv_path, test_csv_path, train_path):
    """Load training and test data"""
    logger.info("Loading training data...")
    train_df = pd.read_csv(train_csv_path)
    
    logger.info("Loading test data...")
    test_df = pd.read_csv(test_csv_path)
    
    # Create label mapping
    unique_labels = sorted(set(train_df['label'].unique()) | set(test_df['label'].unique()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    
    # Load training images
    logger.info("Loading training images...")
    train_images = []
    train_labels = []
    for idx, row in train_df.iterrows():
        img_path = Path(train_path) / row['filename']
        img = load_and_preprocess_image(img_path)
        if img is not None:
            train_images.append(img)
            train_labels.append(label_map[row['label']])
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    # Load test images
    logger.info("Loading test images...")
    test_images = []
    test_labels = []
    for idx, row in test_df.iterrows():
        img_path = Path(train_path) / row['filename']
        img = load_and_preprocess_image(img_path)
        if img is not None:
            test_images.append(img)
            test_labels.append(label_map[row['label']])
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    logger.info(f"[OK] Training: {len(train_images)} images, {len(unique_labels)} classes")
    logger.info(f"[OK] Testing: {len(test_images)} images")
    logger.info(f"Classes: {list(unique_labels)}")
    
    # Check class distribution
    unique, counts = np.unique(train_labels, return_counts=True)
    logger.info("\nClass Distribution:")
    for label_idx, count in zip(unique, counts):
        logger.info(f"  {reverse_label_map[label_idx]}: {count} images")
    
    return train_images, train_labels, test_images, test_labels, label_map, reverse_label_map, unique_labels


def build_model(num_classes, input_shape=(224, 224, 3)):
    """Build MobileNetV3 model with custom head"""
    logger.info("Building MobileNetV3 model...")
    
    base_model = MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Custom head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=5e-4),  # Slightly higher learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built with {num_classes} output classes")
    return model, base_model


def create_balanced_data_generator(images, labels, batch_size=32):
    """Create Keras ImageDataGenerator with BALANCED augmentation"""
    
    # Use Keras built-in ImageDataGenerator (less aggressive)
    train_datagen = ImageDataGenerator(
        rotation_range=15,           # Slight rotation
        width_shift_range=0.1,       # Slight shift
        height_shift_range=0.1,      # Slight shift
        shear_range=0.1,             # Slight shear
        zoom_range=0.1,              # Slight zoom
        brightness_range=[0.9, 1.1], # Slight brightness
        fill_mode='nearest',
        horizontal_flip=False,       # Don't flip pills horizontally
        vertical_flip=False,         # Don't flip pills vertically
    )
    
    # Normalize images
    images_normalized = images / 255.0 if images.max() > 1.0 else images
    labels_cat = to_categorical(labels, num_classes=len(np.unique(labels)))
    
    return train_datagen.flow(images_normalized, labels_cat, batch_size=batch_size, shuffle=True)


def train_with_class_weights(model, train_images, train_labels, test_images, test_labels, 
                             model_path, epochs=50, batch_size=32):
    """Train model with class weights to handle imbalance"""
    
    logger.info("="*60)
    logger.info("TRAINING WITH BALANCED APPROACH")
    logger.info("="*60)
    
    # Normalize training data
    train_images_norm = train_images / 255.0 if train_images.max() > 1.0 else train_images
    test_images_norm = test_images / 255.0 if test_images.max() > 1.0 else test_images
    
    # Create class weights for imbalance
    num_classes = len(np.unique(train_labels))
    class_weights_dict = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_labels),
        train_labels
    )
    class_weights = {i: class_weights_dict[i] for i in range(num_classes)}
    
    logger.info("\nClass Weights (for handling imbalance):")
    for class_idx, weight in class_weights.items():
        logger.info(f"  Class {class_idx}: {weight:.3f}")
    
    # Create data generator
    train_generator = create_balanced_data_generator(train_images, train_labels, batch_size=batch_size)
    
    # Prepare test data
    test_labels_cat = to_categorical(test_labels, num_classes=num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(model_path), monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train
    steps_per_epoch = len(train_images) // batch_size
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(test_images_norm, test_labels_cat),
        callbacks=callbacks,
        class_weight=class_weights,  # Use class weights
        verbose=1
    )
    
    return history


def evaluate_model(model, test_images, test_labels, reverse_label_map):
    """Evaluate model on test set"""
    logger.info("\n" + "="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    test_images_norm = test_images / 255.0 if test_images.max() > 1.0 else test_images
    
    predictions = model.predict(test_images_norm, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    pred_confidence = np.max(predictions, axis=1)
    
    accuracy = accuracy_score(test_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, pred_labels, average='weighted'
    )
    
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    
    # Show per-class metrics
    logger.info("\nPer-Class Metrics:")
    for class_idx in np.unique(test_labels):
        class_mask = test_labels == class_idx
        class_accuracy = accuracy_score(test_labels[class_mask], pred_labels[class_mask])
        class_conf = np.mean(pred_confidence[class_mask])
        logger.info(f"  {reverse_label_map[class_idx]}: Accuracy={class_accuracy:.2%}, Avg Confidence={class_conf:.2%}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }


def main():
    """Main training pipeline"""
    paths = get_data_paths()
    
    # Load data
    train_images, train_labels, test_images, test_labels, label_map, reverse_label_map, unique_labels = load_data(
        paths['train_csv'],
        paths['test_csv'],
        paths['train_path']
    )
    
    # Build model
    model, base_model = build_model(num_classes=len(unique_labels))
    
    # Train with class weights and balanced augmentation
    history = train_with_class_weights(
        model,
        train_images,
        train_labels,
        test_images,
        test_labels,
        paths['model_path'],
        epochs=60,
        batch_size=32
    )
    
    # Evaluate
    metrics = evaluate_model(model, test_images, test_labels, reverse_label_map)
    
    # Save metadata
    metadata = {
        'model_type': 'MobileNetV3Large',
        'training_date': datetime.now().isoformat(),
        'num_classes': len(unique_labels),
        'classes': list(unique_labels),
        'label_map': label_map,
        'metrics': metrics,
        'training_approach': {
            'augmentation': 'Balanced (rotation, shift, zoom, brightness)',
            'augmentation_intensity': 'Moderate (preserves class information)',
            'class_weights': 'Applied for imbalance handling',
            'note': 'Less aggressive than v1, focuses on learning pill classes'
        }
    }
    
    with open(paths['model_metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n[OK] Model saved to {paths['model_path']}")
    logger.info(f"[OK] Metadata saved to {paths['model_metadata']}")
    logger.info("\n[SUCCESS] Training complete! Model should predict pill classes correctly now.")


if __name__ == '__main__':
    main()
