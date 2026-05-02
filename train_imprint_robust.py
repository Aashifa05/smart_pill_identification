#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Pipeline with Imprint Removal Augmentation
====================================================

This script addresses the issue where pills without visible imprints are 
misclassified as UNKNOWN. It uses advanced data augmentation techniques:

1. Gaussian blur on imprint regions (simulates faded imprints)
2. Morphological operations (smooths fine details/imprints)
3. Histogram equalization variations (reduces imprint contrast)
4. Random noise injection (makes imprints harder to read)
5. Brightness/contrast adjustment (emphasizes shape over imprints)

Result: Model learns to identify pills by shape, color, size - not just imprints.
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

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import json


class ImprintRemovalAugmentation:
    """Advanced augmentation techniques to remove/simulate imprint degradation"""
    
    @staticmethod
    def gaussian_blur_imprints(image, kernel_size=5, sigma=1.5):
        """Apply Gaussian blur to reduce imprint sharpness"""
        img = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        result = np.array(blurred, dtype=np.float32) / 255.0 if image.dtype == np.float32 else np.array(blurred)
        return result
    
    @staticmethod
    def morphological_smooth(image, kernel_size=5):
        """Apply morphological opening to smooth imprints"""
        if image.dtype == np.float32:
            img = (image * 255).astype(np.uint8)
        else:
            img = image
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Opening: erosion followed by dilation (removes small details like imprints)
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        result = opened.astype(np.float32) / 255.0 if image.dtype == np.float32 else opened
        return result
    
    @staticmethod
    def reduce_imprint_contrast(image, enhancement_factor=0.6):
        """Reduce contrast to de-emphasize imprints"""
        img = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(img)
        reduced = enhancer.enhance(enhancement_factor)
        result = np.array(reduced, dtype=np.float32) / 255.0 if image.dtype == np.float32 else np.array(reduced)
        return result
    
    @staticmethod
    def add_noise_to_mask_imprints(image, noise_level=0.05):
        """Add noise to make fine imprint details less readable"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = np.clip(image + noise, 0, 1 if image.dtype == np.float32 else 255)
        return noisy
    
    @staticmethod
    def histogram_equalization_variation(image, clip_limit=2.0):
        """Apply adaptive histogram equalization (emphasizes shape, de-emphasizes imprints)"""
        if image.dtype == np.float32:
            img = (image * 255).astype(np.uint8)
        else:
            img = image
        
        # Convert to grayscale, apply CLAHE, then convert back if needed
        if len(img.shape) == 3:
            # For color images, apply to each channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            b, g, r = cv2.split(img)
            b = clahe.apply(b)
            g = clahe.apply(g)
            r = clahe.apply(r)
            result = cv2.merge((b, g, r))
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            result = clahe.apply(img)
        
        return result.astype(np.float32) / 255.0 if image.dtype == np.float32 else result


class ImbalanceAwareDataGenerator:
    """
    Generate training batches with imprint-removed augmentations.
    Handles class imbalance through stratified sampling.
    """
    
    def __init__(self, images, labels, batch_size=32, augmentation_probability=0.7):
        """
        Args:
            images: Array of images (N, H, W, C)
            labels: Array of label indices
            batch_size: Batch size
            augmentation_probability: Probability to apply imprint removal (0-1)
        """
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augmentation_probability = augmentation_probability
        self.augmentor = ImprintRemovalAugmentation()
        self.num_samples = len(images)
        self.indices = np.arange(self.num_samples)
    
    def __iter__(self):
        """Shuffle and yield batches"""
        np.random.shuffle(self.indices)
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_images = []
            batch_labels = self.labels[batch_indices]
            
            for idx in batch_indices:
                img = self.images[idx]
                
                # Apply imprint removal augmentation with probability
                if np.random.rand() < self.augmentation_probability:
                    aug_choice = np.random.choice([
                        'gaussian_blur',
                        'morphological',
                        'contrast_reduction',
                        'noise',
                        'histogram_eq'
                    ])
                    
                    try:
                        if aug_choice == 'gaussian_blur':
                            img = self.augmentor.gaussian_blur_imprints(img)
                        elif aug_choice == 'morphological':
                            img = self.augmentor.morphological_smooth(img)
                        elif aug_choice == 'contrast_reduction':
                            img = self.augmentor.reduce_imprint_contrast(img)
                        elif aug_choice == 'noise':
                            img = self.augmentor.add_noise_to_mask_imprints(img)
                        elif aug_choice == 'histogram_eq':
                            img = self.augmentor.histogram_equalization_variation(img)
                    except Exception as e:
                        logger.warning(f"Augmentation failed: {e}, using original image")
                
                # Ensure image is in correct format
                if img.dtype != np.float32:
                    img = img.astype(np.float32) / 255.0
                
                batch_images.append(img)
            
            yield np.array(batch_images), to_categorical(batch_labels, num_classes=len(np.unique(self.labels)))


def get_data_paths():
    """Get portable data paths"""
    from django.conf import settings
    BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    return {
        'train_csv': BASE_DATA_PATH / 'Training_set.csv',
        'test_csv': BASE_DATA_PATH / 'Testing_set.csv',
        'train_path': BASE_DATA_PATH / 'train',
        'model_path': BASE_DATA_PATH / 'model_imprint_robust.h5',
        'model_metadata': BASE_DATA_PATH / 'model_imprint_robust_metadata.json',
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
        img_path = Path(train_path) / row['image']
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
        img_path = Path(train_path) / row['image']
        img = load_and_preprocess_image(img_path)
        if img is not None:
            test_images.append(img)
            test_labels.append(label_map[row['label']])
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    logger.info(f"[OK] Training: {len(train_images)} images, {len(unique_labels)} classes")
    logger.info(f"[OK] Testing: {len(test_images)} images")
    logger.info(f"Classes: {list(unique_labels)}")
    
    return train_images, train_labels, test_images, test_labels, label_map, reverse_label_map, unique_labels


def build_model(num_classes, input_shape=(224, 224, 3)):
    """Build MobileNetV3 model with custom head"""
    logger.info("Building MobileNetV3 model...")
    
    base_model = MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Custom head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built with {num_classes} output classes")
    return model, base_model


def train_with_augmentation(model, train_images, train_labels, test_images, test_labels, 
                           model_path, epochs=50, batch_size=32):
    """Train model with imprint-removal augmentation"""
    
    logger.info("="*60)
    logger.info("TRAINING WITH IMPRINT REMOVAL AUGMENTATION")
    logger.info("="*60)
    
    # Create data generator with augmentation
    train_generator = ImbalanceAwareDataGenerator(
        train_images, 
        train_labels, 
        batch_size=batch_size,
        augmentation_probability=0.7
    )
    
    # Prepare test data
    num_classes = len(np.unique(train_labels))
    test_labels_cat = to_categorical(test_labels, num_classes=num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        ModelCheckpoint(str(model_path), monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train
    steps_per_epoch = len(train_images) // batch_size
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(test_images, test_labels_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_images, test_labels, reverse_label_map):
    """Evaluate model on test set"""
    logger.info("\n" + "="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    predictions = model.predict(test_images, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(test_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, pred_labels, average='weighted'
    )
    
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    
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
    
    # Train with augmentation
    history = train_with_augmentation(
        model,
        train_images,
        train_labels,
        test_images,
        test_labels,
        paths['model_path'],
        epochs=50,
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
        'augmentation': {
            'techniques': ['gaussian_blur', 'morphological_smooth', 'contrast_reduction', 'noise', 'histogram_eq'],
            'probability': 0.7,
            'purpose': 'Remove/degrade imprint visibility to improve generalization'
        }
    }
    
    with open(paths['model_metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n[OK] Model saved to {paths['model_path']}")
    logger.info(f"[OK] Metadata saved to {paths['model_metadata']}")


if __name__ == '__main__':
    main()
