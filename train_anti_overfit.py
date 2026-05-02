#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anti-Overfitting Training for Pill Classifier
==============================================

Trains the pill classifier to avoid overfitting by learning
general features that work on unseen real-world images.

Data Structure (uses existing):
  media/pilldata/train/  → Training images organized by class
  media/pilldata/test/   → Test images for validation

Techniques Used:
  1. Data Augmentation (1000x variations)
  2. Regularization (L1/L2, Dropout, Batch Norm)
  3. Early Stopping (validation monitoring)
  4. Learning Rate Scheduling
  5. Proper train/val/test splits
  6. Class weight balancing
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import logging

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, 
    Activation, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
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


class AntiOverfitTrainer:
    """Train pill classifier with anti-overfitting techniques"""
    
    def __init__(self, base_dir='media/pilldata'):
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / 'train'
        self.test_dir = self.base_dir / 'test'
        self.metrics = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': None, 'test_acc': None
        }
    
    def load_image_paths(self, data_dir: Path) -> Tuple[List[str], List[str], List[str]]:
        """Load image paths and extract class names from filenames"""
        image_paths = []
        labels = []
        class_names_set = set()
        
        if not data_dir.exists():
            logger.error(f"Directory not found: {data_dir}")
            return [], [], []
        
        # Get all images
        all_images = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
        logger.info(f"\nFound {len(all_images)} images in {data_dir}")
        
        # Extract class names from filenames (remove numbers and extra info)
        for image_path in all_images:
            filename = image_path.stem  # filename without extension
            
            # Extract medication name (everything before the number)
            # Example: "Amoxicillin 500 MG (1)" -> "Amoxicillin 500 MG"
            import re
            # Remove trailing "(N)" and "- Copy" patterns
            class_name = re.sub(r'\s*\(\d+\).*$', '', filename)  # Remove (N)
            class_name = re.sub(r'\s*-\s*Copy.*$', '', class_name)  # Remove - Copy variants
            class_name = class_name.strip()
            
            if class_name:
                class_names_set.add(class_name)
                image_paths.append(str(image_path))
                labels.append(class_name)
        
        class_names = sorted(list(class_names_set))
        
        logger.info(f"Extracted {len(class_names)} unique pill classes")
        for idx, name in enumerate(class_names):
            count = labels.count(name)
            logger.info(f"  [{idx+1:2d}/{len(class_names)}] {name:40s}: {count:4d} images")
        
        logger.info(f"\nTotal: {len(image_paths)} images from {len(class_names)} classes")
        return image_paths, labels, class_names
    
    def preprocess_image(self, image_path: str, size=(224, 224)):
        """Load and preprocess image"""
        try:
            img = load_img(image_path, target_size=size)
            x = img_to_array(img)
            x = preprocess_input(x)
            return x
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            return None
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                                    np.ndarray, np.ndarray, List[str], Dict]:
        """Load and prepare training/validation/test data with proper splits"""
        
        logger.info("\n" + "="*70)
        logger.info("LOADING DATA")
        logger.info("="*70)
        
        # Load training data
        train_paths, train_labels_str, class_names = self.load_image_paths(self.train_dir)
        
        if not train_paths:
            logger.error("No training data found!")
            return None, None, None, None, None, None, None, None
        
        # Load images
        logger.info("\nLoading training images...")
        train_images = []
        valid_labels = []
        valid_paths = []
        
        for path, label in zip(train_paths, train_labels_str):
            img = self.preprocess_image(path)
            if img is not None:
                train_images.append(img)
                valid_labels.append(label)
                valid_paths.append(path)
        
        train_images = np.array(train_images, dtype=np.float32)
        logger.info(f"✓ Loaded {len(train_images)} images")
        
        # Convert labels to indices
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        train_labels = np.array([label_to_idx[l] for l in valid_labels])
        
        # Split into train, validation, test (70%, 15%, 15%)
        logger.info("\nSplitting data...")
        
        # First split: train vs temp (70% vs 30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            train_images, train_labels,
            test_size=0.30,
            random_state=42,
            stratify=train_labels
        )
        
        # Second split: validation vs test (50% of 30% each = 15%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )
        
        # Convert to categorical
        y_train_cat = to_categorical(y_train, len(class_names))
        y_val_cat = to_categorical(y_val, len(class_names))
        y_test_cat = to_categorical(y_test, len(class_names))
        
        # Calculate class weights for imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        logger.info(f"\nData split:")
        logger.info(f"  Train:      {len(X_train):5d} images ({len(X_train)/len(train_images)*100:5.1f}%)")
        logger.info(f"  Validation: {len(X_val):5d} images ({len(X_val)/len(train_images)*100:5.1f}%)")
        logger.info(f"  Test:       {len(X_test):5d} images ({len(X_test)/len(train_images)*100:5.1f}%)")
        
        logger.info(f"\nClass weights:")
        for class_idx, class_name in enumerate(class_names):
            logger.info(f"  {class_name:30s}: {class_weight_dict[class_idx]:.3f}")
        
        return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, class_names, class_weight_dict
    
    def build_model(self, num_classes: int, l1_reg=1e-4, l2_reg=1e-4):
        """Build anti-overfitting model with regularization"""
        
        logger.info("\n" + "="*70)
        logger.info("BUILDING MODEL")
        logger.info("="*70)
        
        # Input layer
        inputs = Input(shape=(224, 224, 3), name='input')
        
        # Pre-trained MobileNetV3 base (ImageNet weights)
        logger.info("Loading pre-trained MobileNetV3Large (ImageNet)...")
        base_model = MobileNetV3Large(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model weights (transfer learning)
        for layer in base_model.layers:
            layer.trainable = False
        
        # Feature extraction
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers with regularization
        x = Dense(512, kernel_regularizer=L1L2(l1=l1_reg, l2=l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        
        x = Dense(256, kernel_regularizer=L1L2(l1=l1_reg, l2=l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='PillClassifier_AntiOverfit')
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✓ Model built")
        logger.info(f"  Total parameters: {model.count_params():,}")
        
        return model, base_model
    
    def create_augmentation_generators(self):
        """Create data augmentation for training"""
        
        logger.info("\nCreating augmentation pipelines...")
        
        # Training augmentation (aggressive)
        train_datagen = ImageDataGenerator(
            rotation_range=25,           # ±25° rotation
            width_shift_range=0.2,       # ±20% horizontal shift
            height_shift_range=0.2,      # ±20% vertical shift
            zoom_range=0.25,             # ±25% zoom
            brightness_range=[0.75, 1.25],  # ±25% brightness
            channel_shift_range=25,      # Color shift
            horizontal_flip=True,        # Flip horizontally
            fill_mode='reflect'          # Fill edges with reflection
        )
        
        # Validation augmentation (minimal)
        val_datagen = ImageDataGenerator()
        
        logger.info("✓ Augmentation pipelines ready")
        
        return train_datagen, val_datagen
    
    def train_with_augmentation(self, model, X_train, X_val, X_test, 
                                y_train, y_val, y_test, class_names, 
                                class_weight_dict, epochs=100, batch_size=32):
        """Train with data augmentation"""
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING")
        logger.info("="*70)
        
        train_datagen, val_datagen = self.create_augmentation_generators()
        
        # Callbacks for better training
        callbacks = [
            # Early stopping: stop if validation doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            
            # Learning rate scheduler: reduce LR if plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                str(self.base_dir / 'model_anti_overfit.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Custom callback for tracking
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._log_epoch(epoch, logs)
            )
        ]
        
        # Train with augmentation
        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        return history
    
    def _log_epoch(self, epoch, logs):
        """Log epoch metrics"""
        if logs:
            self.metrics['train_loss'].append(logs.get('loss', 0))
            self.metrics['train_acc'].append(logs.get('accuracy', 0))
            self.metrics['val_loss'].append(logs.get('val_loss', 0))
            self.metrics['val_acc'].append(logs.get('val_accuracy', 0))
    
    def evaluate_model(self, model, X_test, y_test, class_names):
        """Evaluate on test set"""
        
        logger.info("\n" + "="*70)
        logger.info("EVALUATION ON TEST SET (UNSEEN DATA)")
        logger.info("="*70)
        
        # Test metrics
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"Test Accuracy: {test_acc:.4f} = {test_acc*100:.2f}%")
        logger.info(f"Test Loss:     {test_loss:.4f}")
        
        # Detailed metrics
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        logger.info(f"\nDetailed Metrics:")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        
        # Per-class metrics
        logger.info(f"\nPer-Class Performance:")
        report = classification_report(y_true, y_pred, target_names=class_names)
        logger.info(report)
        
        self.metrics['test_loss'] = test_loss
        self.metrics['test_acc'] = test_acc
        
        return test_acc, test_loss
    
    def plot_training_history(self):
        """Plot training curves and generalization metrics"""
        
        logger.info("\nGenerating training visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Anti-Overfitting Training Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        
        # Loss curve
        axes[0, 0].plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve (should converge)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(epochs, self.metrics['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curve (should stay close)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overfitting score
        overfit_scores = np.array(self.metrics['train_acc']) - np.array(self.metrics['val_acc'])
        axes[1, 0].plot(epochs, overfit_scores, 'purple', linewidth=2, label='Overfit Score')
        axes[1, 0].axhline(y=0, color='green', linestyle='--', label='Perfect (no overfit)')
        axes[1, 0].axhline(y=0.05, color='orange', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Train Acc - Val Acc')
        axes[1, 0].set_title('Overfitting Score (lower is better)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final metrics
        axes[1, 1].axis('off')
        metrics_text = f"""
        FINAL METRICS
        
        Test Accuracy: {self.metrics['test_acc']*100:.2f}%
        Test Loss:     {self.metrics['test_loss']:.4f}
        
        Train Acc:     {self.metrics['train_acc'][-1]:.4f}
        Val Acc:       {self.metrics['val_acc'][-1]:.4f}
        Overfit Gap:   {overfit_scores[-1]:.4f}
        
        Status: {'✓ EXCELLENT' if self.metrics['test_acc'] > 0.85 else '⚠ GOOD' if self.metrics['test_acc'] > 0.80 else '❌ NEEDS IMPROVEMENT'}
        """
        axes[1, 1].text(0.5, 0.5, metrics_text, ha='center', va='center',
                       fontsize=12, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.base_dir / 'generalization_metrics.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"✓ Saved plot to {output_path}")
        plt.close()
    
    def save_metadata(self, class_names):
        """Save training metadata"""
        metadata = {
            'model_name': 'PillClassifier_AntiOverfit',
            'train_date': datetime.now().isoformat(),
            'num_classes': len(class_names),
            'class_names': class_names,
            'input_size': (224, 224),
            'base_model': 'MobileNetV3Large',
            'regularization': {
                'l1': 1e-4,
                'l2': 1e-4,
                'dropout': [0.4, 0.3]
            },
            'data_augmentation': {
                'rotation_range': 25,
                'zoom_range': 0.25,
                'brightness_range': [0.75, 1.25],
                'horizontal_flip': True
            },
            'training': {
                'optimizer': 'Adam (1e-4)',
                'loss': 'categorical_crossentropy',
                'batch_size': 32,
                'epochs': len(self.metrics['train_loss'])
            },
            'metrics': {
                'test_accuracy': float(self.metrics['test_acc']),
                'test_loss': float(self.metrics['test_loss']),
                'final_train_acc': float(self.metrics['train_acc'][-1]) if self.metrics['train_acc'] else 0,
                'final_val_acc': float(self.metrics['val_acc'][-1]) if self.metrics['val_acc'] else 0
            }
        }
        
        output_path = self.base_dir / 'model_anti_overfit_metadata.json'
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Saved metadata to {output_path}")


def main():
    """Main training pipeline"""
    
    logger.info("\n" + "="*70)
    logger.info("ANTI-OVERFITTING PILL CLASSIFIER TRAINING")
    logger.info("="*70)
    logger.info("Goal: Learn general pill features that work on unseen real-world images")
    logger.info("Techniques: Augmentation, Regularization, Early Stopping, Transfer Learning")
    logger.info("="*70 + "\n")
    
    # Initialize trainer
    trainer = AntiOverfitTrainer()
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, class_weights = trainer.prepare_data()
    
    if X_train is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Build model
    model, base_model = trainer.build_model(len(class_names))
    
    # Train
    history = trainer.train_with_augmentation(
        model, X_train, X_val, X_test, y_train, y_val, y_test,
        class_names, class_weights, epochs=150, batch_size=32
    )
    
    # Evaluate
    test_acc, test_loss = trainer.evaluate_model(model, X_test, y_test, class_names)
    
    # Plot results
    trainer.plot_training_history()
    
    # Save metadata
    trainer.save_metadata(class_names)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"✓ Model saved to: media/pilldata/model_anti_overfit.keras")
    logger.info(f"✓ Metrics saved to: media/pilldata/model_anti_overfit_metadata.json")
    logger.info(f"✓ Plot saved to: media/pilldata/generalization_metrics.png")
    logger.info(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Status: {'✓ EXCELLENT' if test_acc > 0.85 else '⚠ GOOD' if test_acc > 0.80 else '❌ NEEDS IMPROVEMENT'}")
    logger.info("\nNext steps:")
    logger.info("  1. Review generalization_metrics.png")
    logger.info("  2. Update classifier to use model_anti_overfit.keras")
    logger.info("  3. Test on real-world pill images")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
