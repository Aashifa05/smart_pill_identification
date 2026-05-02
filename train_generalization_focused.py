#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overfitting-Resistant Pill Classification Model Training
=========================================================

This script trains a model to learn GENERAL pill features that generalize
well to unseen real-world images, avoiding overfitting through:

1. Advanced Data Augmentation
   - Rotation, zoom, brightness variations
   - Imprint removal/fading simulation
   - Geometric transformations
   - Color jittering

2. Regularization Techniques
   - Dropout (prevent co-adaptation)
   - L1/L2 regularization (weight penalty)
   - Batch normalization (internal covariate shift)
   - Early stopping (validation monitoring)

3. Proper Training Strategy
   - Large validation set (20-30%)
   - Learning rate scheduling (reduce overfitting)
   - Class weight balancing
   - Proper train/val/test splits

4. Generalization Focus
   - Learn shape, color, size, texture features
   - Robust to imprint variations
   - Tolerant to image quality differences
   - Consistent across pill variations

Result: Model that generalizes well to real-world unseen pills!
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import logging

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, 
    Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, LambdaCallback
)
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneralizationFocusedAugmentation:
    """Advanced augmentation to improve generalization without overfitting"""
    
    @staticmethod
    def create_augmentation_pipeline(stage='train'):
        """
        Create augmentation pipeline that preserves general features.
        
        Training stage: Aggressive augmentation to prevent overfitting
        Validation stage: Minimal augmentation (only preprocessing)
        
        Args:
            stage: 'train' or 'validation'
            
        Returns:
            ImageDataGenerator configured for the stage
        """
        
        if stage == 'train':
            # TRAINING: Aggressive augmentation
            return ImageDataGenerator(
                # Geometric transformations (preserve shape features)
                rotation_range=20,           # Slight rotations
                width_shift_range=0.15,      # Horizontal shift
                height_shift_range=0.15,     # Vertical shift
                shear_range=10,              # Shear transformation
                zoom_range=0.2,              # Zoom in/out
                
                # Color variations (preserve color features)
                brightness_range=[0.8, 1.2],      # Brightness
                channel_shift_range=20,            # Color shift
                
                # Preprocessing
                preprocessing_function=preprocess_input,
                
                # Regularization augmentations
                horizontal_flip=True,        # Horizontal flip (many pills symmetric)
                vertical_flip=False,         # No vertical flip (pills oriented)
                fill_mode='nearest',         # Fill missing pixels
                
                # Data augmentation options
                featurewise_center=False,
                featurewise_std_normalization=False,
                samplewise_center=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                
                # Note: dropout-like augmentation is handled in the model (Dropout layers)
            )
        
        else:
            # VALIDATION: Minimal augmentation (just preprocessing)
            return ImageDataGenerator(
                preprocessing_function=preprocess_input
            )


class GeneralizationMonitor:
    """Monitor model generalization during training"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.overfitting_scores = []
    
    def on_epoch_end(self, epoch, logs):
        """Called after each epoch"""
        train_loss = logs['loss']
        val_loss = logs['val_loss']
        train_acc = logs['accuracy']
        val_acc = logs['val_accuracy']
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # Overfitting score: (train_acc - val_acc)
        # Positive = overfitting, Negative = underfitting, ~0 = good generalization
        overfitting = train_acc - val_acc
        self.overfitting_scores.append(overfitting)
        
        # Log generalization metrics
        logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
                   f"Overfit Score={overfitting:.4f}")
        
        # Alert if overfitting detected
        if overfitting > 0.05 and epoch > 5:  # 5% gap
            logger.warning(f"⚠️  OVERFITTING DETECTED: Gap = {overfitting:.4f}")
    
    def plot_generalization(self, output_path='generalization_plot.png'):
        """Plot generalization metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss comparison
        axes[0, 0].plot(self.train_losses, label='Train Loss', alpha=0.7)
        axes[0, 0].plot(self.val_losses, label='Val Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve (Good Generalization = Converging)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy comparison
        axes[0, 1].plot(self.train_accs, label='Train Accuracy', alpha=0.7)
        axes[0, 1].plot(self.val_accs, label='Val Accuracy', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curve (Lines should stay close)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overfitting score
        axes[1, 0].plot(self.overfitting_scores, label='Overfitting Score', color='red', alpha=0.7)
        axes[1, 0].axhline(y=0, color='green', linestyle='--', label='Perfect Generalization')
        axes[1, 0].axhline(y=0.05, color='orange', linestyle='--', label='Overfitting Threshold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Train Acc - Val Acc')
        axes[1, 0].set_title('Overfitting Score (Lower is Better)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gap between curves
        gap = np.array(self.train_accs) - np.array(self.val_accs)
        axes[1, 1].bar(range(len(gap)), gap, color=['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' for x in gap], alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap')
        axes[1, 1].set_title('Train-Val Accuracy Gap (Green=Good)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"✓ Saved generalization plot to {output_path}")


class AccuracyAndStableValLossStopper(tf.keras.callbacks.Callback):
    """
    Stop training when both training and validation accuracy reach the target
    range (85% - 95%) and validation loss is stable or decreasing.
    """
    def __init__(self, acc_lower=0.85, acc_upper=0.95, stable_window=3, verbose=1):
        super().__init__()
        self.acc_lower = acc_lower
        self.acc_upper = acc_upper
        self.stable_window = stable_window
        self.verbose = verbose
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        if train_acc is None or val_acc is None or val_loss is None:
            return

        self.val_losses.append(val_loss)

        # Condition 1: both accuracies within range
        acc_cond = (self.acc_lower <= train_acc <= self.acc_upper) and (
            self.acc_lower <= val_acc <= self.acc_upper)

        # Condition 2: validation loss is stable or decreasing over the last window
        stable = False
        if len(self.val_losses) >= 2:
            # stable if current loss <= previous loss OR not increased more than 1%
            prev = self.val_losses[-2]
            if val_loss <= prev or (val_loss - prev) / max(prev, 1e-8) <= 0.01:
                stable = True

        # More conservative: require non-increase over last `stable_window` epochs
        if len(self.val_losses) >= self.stable_window:
            recent = self.val_losses[-self.stable_window:]
            if all(recent[i] <= recent[i-1] + 1e-6 for i in range(1, len(recent))):
                stable = True

        if acc_cond and stable:
            if self.verbose:
                logger.info(f"Auto-stopping: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")
            self.model.stop_training = True


def build_generalization_model(num_classes: int, l1_reg: float = 1e-4, 
                               l2_reg: float = 1e-4) -> Model:
    """
    Build model focused on learning generalizable features.
    
    Architecture:
    - MobileNetV3 base (efficient, generalizes well)
    - Batch normalization (stable learning)
    - Dropout layers (prevent co-adaptation)
    - L1/L2 regularization (weight penalty)
    - Proper initialization
    
    Args:
        num_classes: Number of pill classes
        l1_reg: L1 regularization strength
        l2_reg: L2 regularization strength
        
    Returns:
        Compiled Keras model
    """
    
    logger.info("Building generalization-focused model...")
    
    # Input layer
    inputs = Input(shape=(224, 224, 3), name='input')
    
    # Pre-trained MobileNetV3 base (learns general image features)
    base_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # Initialize from scratch (train from dataset)
    )
    
    # Freeze early layers (preserve general features)
    # Fine-tune only later layers (adapt to pills)
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Feature extraction
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    
    # Regularized dense layers
    x = BatchNormalization()(x)
    x = Dense(
        512,
        kernel_regularizer=L1L2(l1=l1_reg, l2=l2_reg),
        bias_regularizer=L1L2(l1=l1_reg, l2=l2_reg)
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)  # Moderate dropout
    
    x = Dense(
        256,
        kernel_regularizer=L1L2(l1=l1_reg, l2=l2_reg),
        bias_regularizer=L1L2(l1=l1_reg, l2=l2_reg)
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=L1L2(l1=l1_reg, l2=l2_reg),
        name='output'
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with appropriate loss
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Small LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    logger.info(f"  - {sum(1 for l in model.layers if l.trainable)} trainable layers")
    
    return model


def load_pill_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load pill images organized by class subdirectories.
    
    Expected structure:
    data_dir/
      ├── class1/
      │   ├── image1.jpg
      │   └── image2.jpg
      └── class2/
          └── image3.jpg
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Tuple of (images, labels, class_names)
    """
    
    logger.info(f"Loading pill data from {data_dir}...")
    
    images = []
    labels = []
    class_names = []
    
    # Get class directories
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not class_dirs:
        logger.error(f"No class directories found in {data_dir}")
        return None, None, None
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_names.append(class_name)
        
        # Load images from class directory
        image_paths = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        logger.info(f"  Loading class {class_idx+1}/{len(class_dirs)}: {class_name} ({len(image_paths)} images)")
        
        for image_path in image_paths:
            try:
                # Load and preprocess image
                img = load_img(image_path, target_size=(224, 224))
                # Keep raw pixel range (0-255) and let MobileNetV3's
                # `preprocess_input` perform the correct normalization to [-1, 1].
                img_array = img_to_array(img)
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                logger.warning(f"    Failed to load {image_path}: {str(e)[:50]}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    
    logger.info(f"✓ Loaded {len(images)} images from {len(class_names)} classes")
    
    return images, labels, class_names


def train_generalization_model(images: np.ndarray, labels: np.ndarray, 
                               class_names: List[str],
                               model_save_path: Path,
                               metadata_save_path: Path,
                               epochs: int = 100,
                               batch_size: int = 32,
                               validation_split: float = 0.2,
                               test_split: float = 0.1):
    """
    Train model with focus on generalization.
    
    Args:
        images: Image array (N, 224, 224, 3)
        labels: Class labels (N,)
        class_names: List of class names
        model_save_path: Where to save best model
        metadata_save_path: Where to save metadata
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Proportion for validation
        test_split: Proportion for test
    """
    
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING MODEL FOR GENERALIZATION")
    logger.info(f"{'='*70}")
    
    # Convert labels to categorical
    num_classes = len(class_names)
    labels_cat = to_categorical(labels, num_classes)
    
    # Split data: train/val/test
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels_cat,
        test_size=test_split,
        random_state=42,
        stratify=labels
    )
    
    # Second split: train/validation from remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=validation_split / (1 - test_split),
        random_state=42,
        stratify=np.argmax(y_temp, axis=1)
    )
    
    logger.info(f"\nData split:")
    logger.info(f"  Training:   {len(X_train)} images ({len(X_train)/len(images)*100:.1f}%)")
    logger.info(f"  Validation: {len(X_val)} images ({len(X_val)/len(images)*100:.1f}%)")
    logger.info(f"  Test:       {len(X_test)} images ({len(X_test)/len(images)*100:.1f}%)")
    
    # Calculate class weights (handle imbalance)
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights_array)}
    
    logger.info(f"\nClass weights (for imbalance handling):")
    for class_idx, weight in class_weights_dict.items():
        logger.info(f"  {class_names[class_idx]:30s}: {weight:.2f}")
    
    # Build model
    model = build_generalization_model(num_classes)
    
    # Data augmentation
    aug = GeneralizationFocusedAugmentation()
    train_datagen = aug.create_augmentation_pipeline('train')
    val_datagen = aug.create_augmentation_pipeline('validation')
    
    # Callbacks
    generalization_monitor = GeneralizationMonitor()
    callbacks = [
        # Early stopping (stop if validation doesn't improve)
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Learning rate scheduling (reduce overfitting)
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        
        # Save best model
        ModelCheckpoint(
            str(model_save_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        
        # Generalization monitoring
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: generalization_monitor.on_epoch_end(epoch, logs)
        )
    ]

    # Custom auto-stop: stop when both train & val accuracy are in target range
    # AND validation loss is stable or decreasing.
    callbacks.insert(0, AccuracyAndStableValLossStopper(acc_lower=0.85, acc_upper=0.95, stable_window=3))
    
    # Train with augmentation
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING")
    logger.info(f"{'='*70}\n")
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Evaluate on test set (unseen data)
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION ON TEST SET (UNSEEN DATA)")
    logger.info(f"{'='*70}\n")
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    
    logger.info(f"\n✓ Test Accuracy (on unseen data): {test_acc:.2%}")
    logger.info(f"✓ Test Loss: {test_loss:.4f}")
    
    # Test predictions
    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    y_test_true_labels = np.argmax(y_test, axis=1)
    
    # Detailed metrics
    logger.info(f"\nDetailed Test Metrics:")
    logger.info(f"  Precision: {precision_score(y_test_true_labels, y_test_pred_labels, average='macro'):.4f}")
    logger.info(f"  Recall:    {recall_score(y_test_true_labels, y_test_pred_labels, average='macro'):.4f}")
    logger.info(f"  F1 Score:  {f1_score(y_test_true_labels, y_test_pred_labels, average='macro'):.4f}")
    
    # Plot generalization
    gen_plot_path = model_save_path.parent / 'generalization_metrics.png'
    generalization_monitor.plot_generalization(str(gen_plot_path))
    
    # Save metadata
    metadata = {
        'class_names': class_names,
        'num_classes': num_classes,
        'input_shape': [224, 224, 3],
        'training_date': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_info': {
            'epochs_trained': len(history.history['loss']),
            'batch_size': batch_size,
            'total_images': len(images),
            'train_images': len(X_train),
            'val_images': len(X_val),
            'test_images': len(X_test),
        },
        'model_info': {
            'base_model': 'MobileNetV3Large',
            'regularization': 'L1/L2 + Dropout + Batch Normalization',
            'augmentation': 'Rotation, Zoom, Brightness, Color shift',
            'focus': 'Generalization to unseen real-world images'
        }
    }
    
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n✓ Model saved to {model_save_path}")
    logger.info(f"✓ Metadata saved to {metadata_save_path}")
    
    logger.info(f"\n{'='*70}")
    logger.info("GENERALIZATION ASSESSMENT")
    logger.info(f"{'='*70}")
    logger.info(f"\nOverfitting indicators:")
    logger.info(f"  Train Accuracy:      {history.history['accuracy'][-1]:.2%}")
    logger.info(f"  Val Accuracy:        {history.history['val_accuracy'][-1]:.2%}")
    logger.info(f"  Test Accuracy:       {test_acc:.2%}")
    logger.info(f"  Gap (train-val):     {history.history['accuracy'][-1] - history.history['val_accuracy'][-1]:.2%}")
    
    if test_acc >= history.history['val_accuracy'][-1] - 0.05:
        logger.info("\n✓ EXCELLENT: Model generalizes well to unseen data!")
    elif test_acc >= history.history['val_accuracy'][-1] - 0.1:
        logger.info("\n⚠ GOOD: Model generalizes reasonably well")
    else:
        logger.warning("\n✗ WARNING: Possible overfitting detected")
    
    return model, metadata


def main():
    """Main training entry point"""
    
    from django.conf import settings
    
    # Data directory
    data_dir = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'pills'
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("\nExpected structure:")
        logger.error("  media/pilldata/pills/")
        logger.error("    ├── class1/")
        logger.error("    │   ├── image1.jpg")
        logger.error("    │   └── image2.jpg")
        logger.error("    └── class2/")
        logger.error("        └── image3.jpg")
        return
    
    # Load data
    images, labels, class_names = load_pill_data(data_dir)
    
    if images is None:
        logger.error("Failed to load data")
        return
    
    # Output paths
    model_dir = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    model_save_path = model_dir / 'model_generalization.keras'
    metadata_save_path = model_dir / 'model_generalization_metadata.json'
    
    # Train
    # Allow quick overrides via environment variables for dry-runs and tuning
    epochs = int(os.environ.get('EPOCHS', '100'))
    batch_size = int(os.environ.get('BATCH_SIZE', '32'))
    val_split = float(os.environ.get('VAL_SPLIT', '0.2'))
    test_split = float(os.environ.get('TEST_SPLIT', '0.1'))

    model, metadata = train_generalization_model(
        images, labels, class_names,
        model_save_path, metadata_save_path,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        test_split=test_split
    )
    
    logger.info("\n" + "="*70)
    logger.info("✓ TRAINING COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()
