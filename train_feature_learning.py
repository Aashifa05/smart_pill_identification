#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Pill Classifier Training - Feature-Focused
=====================================================

This training pipeline addresses the core issues:

1. MISCLASSIFICATION: Model learns to distinguish pills by multiple features
2. UNSEEN DATA: Features (shape, color, imprints) generalize better than imprint text
3. CONFIDENCE: Training focused on learning discriminative features

Strategy:
---------
1. Feature-based learning (shape, color, imprint, texture)
2. Data augmentation to increase robustness
3. Class balancing to prevent bias
4. Validation on diverse images
5. Early stopping to prevent overfitting

Result: Model learns PILL FEATURES not just imprints
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, GlobalAveragePooling2D, 
    Dropout, Activation, Add, MaxPooling2D, Conv2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import seaborn as sns
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from datetime import datetime
import csv

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedAugmentation:
    """
    Advanced augmentation techniques for robustness.
    Focuses on preserving features while varying appearance.
    """
    
    @staticmethod
    def augment_image(image, augmentation_type='random'):
        """
        Apply augmentation while preserving pill features.
        
        Args:
            image: PIL Image
            augmentation_type: Type of augmentation
            
        Returns:
            Augmented PIL Image
        """
        if augmentation_type == 'random':
            augmentation_type = np.random.choice([
                'brightness', 'contrast', 'rotation', 'blur', 'noise', 'hue'
            ])
        
        if augmentation_type == 'brightness':
            # Brightness variation (0.7x - 1.3x)
            enhancer = ImageEnhance.Brightness(image)
            factor = np.random.uniform(0.7, 1.3)
            return enhancer.enhance(factor)
        
        elif augmentation_type == 'contrast':
            # Contrast variation
            enhancer = ImageEnhance.Contrast(image)
            factor = np.random.uniform(0.8, 1.2)
            return enhancer.enhance(factor)
        
        elif augmentation_type == 'rotation':
            # Small rotation (±15 degrees)
            angle = np.random.uniform(-15, 15)
            return image.rotate(angle, expand=False, resample=Image.BICUBIC)
        
        elif augmentation_type == 'blur':
            # Slight blur (simulates camera focus variation)
            blur_radius = np.random.uniform(0.5, 2.0)
            return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        elif augmentation_type == 'saturation':
            # Color saturation variation
            enhancer = ImageEnhance.Color(image)
            factor = np.random.uniform(0.6, 1.4)
            return enhancer.enhance(factor)
        
        else:
            return image
    
    @staticmethod
    def augment_batch(images, labels, augmentation_per_image=2):
        """
        Create augmented copies of training images.
        Increases dataset diversity while maintaining labels.
        """
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            # Original
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Augmented copies
            for _ in range(augmentation_per_image):
                try:
                    # images are in 0-255 range; convert to uint8 for PIL
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    augmented = AdvancedAugmentation.augment_image(pil_img, 'random')
                    # keep augmented images in 0-255 float range
                    aug_array = np.array(augmented, dtype=np.float32)
                    
                    augmented_images.append(aug_array)
                    augmented_labels.append(label)
                except Exception as e:
                    logger.warning(f"Augmentation failed: {e}")
        
        return np.array(augmented_images), np.array(augmented_labels)

class FeatureLearningModel:
    """
    Build a model optimized for feature learning.
    Uses residual connections and feature extraction layers.
    """
    
    @staticmethod
    def build_model(num_classes, input_shape=(224, 224, 3)):
        """
        Build advanced feature-learning model.
        
        Strategy:
        1. Use MobileNetV3Large as backbone (efficient + powerful)
        2. Add custom feature extraction layers
        3. Multiple classification heads for robustness
        4. Residual connections for better gradient flow
        """
        
        # Input
        inputs = Input(shape=input_shape)
        
        # Backbone: MobileNetV3Large
        mobilenet = MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )

        # Freeze all base MobileNet layers initially
        for layer in mobilenet.layers:
            layer.trainable = False

        # Feature extraction
        x = mobilenet(inputs)

        # Global pooling head: GlobalAveragePooling2D -> Dropout(0.3) -> Dense(num_classes, softmax)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='FeatureLearningModel')
        
        return model
    
    @staticmethod
    def compile_model(model, learning_rate=1e-4):
        """Compile with optimal settings"""
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class DataBalancer:
    """Handle class imbalance in training data"""
    
    @staticmethod
    def get_class_weights(labels):
        """Compute class weights for imbalanced data"""
        num_classes = len(np.unique(labels))
        class_counts = np.bincount(labels.argmax(axis=1) if len(labels.shape) > 1 else labels)
        
        total = class_counts.sum()
        weights = total / (num_classes * class_counts.astype(float))
        
        return {i: w for i, w in enumerate(weights)}
    
    @staticmethod
    def stratified_split(images, labels, test_size=0.2, val_size=0.1):
        """Split data with stratification"""
        label_indices = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_size,
            stratify=label_indices,
            random_state=42
        )
        
        # Second split: train vs val
        label_indices_temp = np.argmax(y_temp, axis=1) if len(y_temp.shape) > 1 else y_temp
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=label_indices_temp,
            random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class EarlyValAccStopper(tf.keras.callbacks.Callback):
    """
    Stop training when validation accuracy reaches a target threshold.
    """
    def __init__(self, target=0.90):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy')
        if val_acc is not None:
            if val_acc >= self.target:
                logger.info(f"Validation accuracy reached {val_acc:.4f} >= {self.target}. Stopping training.")
                self.model.stop_training = True


class PillClassifierTrainer:
    """Complete training pipeline"""
    
    def __init__(self, dataset_dir='media/pilldata/train', 
                 model_output_path='media/pilldata/model_feature_learning_final_best.keras'):
        """Initialize trainer"""
        self.dataset_dir = dataset_dir
        self.model_output_path = model_output_path
        self.metadata_path = model_output_path.replace('.keras', '_metadata.json')
    
    def load_dataset(self):
        """Load training data"""
        logger.info("Loading dataset...")
        
        images = []
        labels = []
        label_map = {}
        class_idx = 0
        
        # Load data from directory
        dataset_path = Path(self.dataset_dir)
        
        # Check if directory exists
        if not dataset_path.exists():
            logger.error(f"❌ Dataset directory not found: {self.dataset_dir}")
            sys.exit(1)
        
        # Find all image files (supporting both flat and nested structure)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_img_files = []
        
        # Check if we have subdirectories (nested) or flat structure
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if subdirs:
            # CASE 1: Nested structure - images in subdirectories by pill class
            logger.info(f"Detected nested directory structure with {len(subdirs)} pill classes")
            for pill_dir in sorted(subdirs):
                pill_name = pill_dir.name
                label_map[pill_name] = class_idx
                class_idx += 1
                
                img_files = []
                for ext in image_extensions:
                    img_files.extend(list(pill_dir.glob(ext)))
                
                logger.info(f"  {pill_name}: {len(img_files)} images")
                for img_path in img_files:
                    all_img_files.append((img_path, pill_name))
        
        else:
            # CASE 2: Flat structure - extract pill name from filename
            logger.info(f"Detected flat directory structure. Extracting pill names from filenames...")
            
            # Find all images in main directory
            for ext in image_extensions:
                all_img_files.extend([(f, None) for f in dataset_path.glob(ext)])
            
            if all_img_files:
                # Extract unique pill names from filenames
                # Handles multiple formats:
                # 1. "Amoxicillin 500 MG (1) - Copy.jpg"  -> Amoxicillin
                # 2. "0002-3228_0_1 - Copy.jpg"  -> 0002-3228 (NDC code)
                
                def extract_pill_name(filename):
                    """Extract pill name from filename with fallback patterns"""
                    # Pattern 1: PillName dosage MG (number)
                    match = re.match(r'([^0-9]*?)\s+\d+', filename)
                    if match:
                        return match.group(1).strip()
                    
                    # Pattern 2: NDC-Code_index or similar
                    # Extract part before first underscore or dash
                    match = re.match(r'([a-zA-Z0-9]+-?[a-zA-Z0-9]*?)_', filename)
                    if match:
                        return match.group(1).strip()
                    
                    # Pattern 3: Just use part before dash
                    match = re.match(r'([a-zA-Z0-9]+-[a-zA-Z0-9]+)', filename)
                    if match:
                        return match.group(1).strip()
                    
                    return None
                
                pill_names = set()
                for img_path, _ in all_img_files:
                    filename = img_path.stem
                    pill_name = extract_pill_name(filename)
                    if pill_name:
                        pill_names.add(pill_name)
                
                # Create label map
                for pill_name in sorted(pill_names):
                    label_map[pill_name] = class_idx
                    class_idx += 1
                
                # Reorganize image files with extracted names
                new_img_files = []
                skipped = 0
                for img_path, _ in all_img_files:
                    filename = img_path.stem
                    pill_name = extract_pill_name(filename)
                    if pill_name:
                        new_img_files.append((img_path, pill_name))
                    else:
                        skipped += 1
                
                if skipped > 0:
                    logger.warning(f"Skipped {skipped} images that could not be classified")
                
                all_img_files = new_img_files
                logger.info(f"Detected {len(pill_names)} unique pill classes from filenames")
                for pill_name in sorted(pill_names):
                    count = sum(1 for _, name in all_img_files if name == pill_name)
                    logger.info(f"  {pill_name}: {count} images")
        
        # Load all images
        for img_path, pill_name in all_img_files:
            if pill_name not in label_map:
                logger.warning(f"Skipping {img_path} - unknown pill class: {pill_name}")
                continue
            
            try:
                img = load_img(str(img_path), target_size=(224, 224))
                # Keep images in 0-255 range (float32). We'll apply
                # mobilenet_v3.preprocess_input later (after augmentation)
                img_array = img_to_array(img).astype(np.float32)
                images.append(img_array)
                pill_idx = label_map[pill_name]
                labels.append(pill_idx)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
        
        total_images = len(images)
        total_classes = len(label_map)
        
        logger.info(f"✓ Loaded {total_images} images from {total_classes} classes")
        
        if total_images == 0:
            logger.error(f"❌ No images were loaded!")
            logger.info(f"Supported formats: jpg, jpeg, png (case-insensitive)")
            sys.exit(1)
        
        images = np.array(images)
        labels = to_categorical(labels, num_classes=len(label_map))

        return images, labels, label_map
    
    def train(self, epochs=20, batch_size=32, augmentation=True):
        """Train the model (up to 20 epochs)"""
        logger.info("=" * 80)
        logger.info("TRAINING: Feature-Learning Pill Classifier")
        logger.info("=" * 80)
        
        # Load data
        images, labels, label_map = self.load_dataset()
        
        # Split data first (prevent augmentation leakage)
        X_train, X_val, X_test, y_train, y_val, y_test = DataBalancer.stratified_split(
            images, labels
        )

        # Augment only the training set to avoid data leakage
        if augmentation:
            logger.info("Augmenting training data (train set only)...")
            X_train, y_train = AdvancedAugmentation.augment_batch(
                X_train, y_train, augmentation_per_image=1
            )
            logger.info(f"After augmentation: {len(X_train)} training images")
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Apply MobileNetV3 preprocessing to all sets (preprocess_input expects 0-255 float images)
        try:
            X_train = preprocess_input(X_train.astype(np.float32))
            X_val = preprocess_input(X_val.astype(np.float32))
            X_test = preprocess_input(X_test.astype(np.float32))
            logger.info("Applied MobileNetV3 preprocess_input to train/val/test sets")
        except Exception as e:
            logger.warning(f"Failed to apply preprocess_input: {e}")
        
        # Verify 20 classes and print class names
        num_classes = len(label_map)
        logger.info("\n" + "=" * 80)
        logger.info(f"PILL CLASSES ({num_classes} total):")
        logger.info("=" * 80)
        for class_name, class_idx in sorted(label_map.items(), key=lambda x: x[1]):
            logger.info(f"  [{class_idx + 1:2d}] {class_name}")
        logger.info("=" * 80 + "\n")
        
        if num_classes != 20:
            logger.warning(f"⚠️  WARNING: Expected 20 classes but found {num_classes}. Proceeding...")
        else:
            logger.info(f"✓ Confirmed: All 20 pill classes loaded successfully")
        
        # Build model
        logger.info("Building model...")
        model = FeatureLearningModel.build_model(len(label_map))
        model = FeatureLearningModel.compile_model(model)
        
        # Callbacks
        # EarlyStopping (monitor='val_loss', patience=5, restore_best_weights=True)
        # ReduceLROnPlateau for learning rate scheduling
        # NOTE: EarlyValAccStopper REMOVED
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_output_path.replace('.keras', '_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Compute class weights
        class_weights = DataBalancer.get_class_weights(y_train)
        
        # Train
        logger.info("Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 80)
        
        # Final training accuracy
        train_pred = model.predict(X_train, verbose=0)
        train_accuracy = accuracy_score(y_train.argmax(axis=1), train_pred.argmax(axis=1))
        logger.info(f"✓ Final Training Accuracy:    {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        
        # Final validation accuracy
        val_pred = model.predict(X_val, verbose=0)
        val_accuracy = accuracy_score(y_val.argmax(axis=1), val_pred.argmax(axis=1))
        logger.info(f"✓ Final Validation Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Test accuracy
        y_pred = model.predict(X_test, verbose=0)
        test_accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        logger.info(f"✓ Test Accuracy:              {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info("=" * 80)
        
        # Per-class metrics
        y_test_labels = y_test.argmax(axis=1)
        y_pred_labels = y_pred.argmax(axis=1)
        
        logger.info("\n" + "=" * 80)
        logger.info("PER-CLASS ACCURACY")
        logger.info("=" * 80)
        
        per_class_accuracies = {}
        for class_idx in range(len(label_map)):
            mask = y_test_labels == class_idx
            if np.sum(mask) > 0:
                class_pred = y_pred_labels[mask]
                class_true = y_test_labels[mask]
                class_accuracy = accuracy_score(class_true, class_pred)
                class_name = [name for name, idx in label_map.items() if idx == class_idx][0]
                per_class_accuracies[class_name] = class_accuracy
                logger.info(f"  {class_name:30s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        logger.info("=" * 80)
        
        report = classification_report(
            y_test_labels, y_pred_labels,
            target_names=[name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
        )
        logger.info("\nDetailed Classification Report:\n" + report)

        # Confusion matrix and plot with professional formatting
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING CONFUSION MATRIX")
        logger.info("=" * 80)
        try:
            import seaborn as sns
            cm = confusion_matrix(y_test_labels, y_pred_labels)
            
            # Get class names in order
            class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
            
            # Create figure with appropriate size for readability
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Use seaborn heatmap for professional appearance
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'}, ax=ax)
            
            # Format labels and title
            ax.set_title('Confusion Matrix - Pill Classification (Final Model)', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
            
            # Rotate x-axis labels for readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('confusion_matrix_final.png', dpi=200, bbox_inches='tight')
            logger.info('✓ Saved confusion matrix to confusion_matrix_final.png')
            plt.close()
        except Exception as e:
            logger.warning(f'Failed to plot confusion matrix: {e}')
        
        # Check if specific pills are predicted
        logger.info("\n" + "=" * 80)
        logger.info("PILL PREDICTION CONFIRMATION")
        logger.info("=" * 80)
        
        target_pills = ['eltrombopag 25 MG', 'sitagliptin 50 MG']
        for target_pill in target_pills:
            pill_found = False
            for name in label_map.keys():
                if target_pill.lower() in name.lower():
                    logger.info(f"✓ {target_pill} is loaded and will be predicted")
                    pill_found = True
                    break
            if not pill_found:
                logger.warning(f"⚠️  {target_pill} NOT found in training classes")
        
        logger.info("=" * 80 + "\n")
        
        # Save model
        logger.info(f"Saving model to {self.model_output_path}")
        model.save(self.model_output_path)
        
        # Save metadata
        metadata = {
            'label_map': label_map,
            'training_accuracy': float(train_accuracy),
            'validation_accuracy': float(val_accuracy),
            'test_accuracy': float(test_accuracy),
            'num_classes': len(label_map),
            'training_date': datetime.now().isoformat(),
            'model_type': 'FeatureLearning',
            'model_architecture': 'MobileNetV3Large (backbone frozen) + GlobalAveragePooling2D + Dropout(0.3) + Dense(20, softmax)',
            'input_shape': [224, 224, 3],
            'optimizer': 'Adam (lr=1e-4)',
            'loss_function': 'categorical_crossentropy',
            'per_class_accuracy': per_class_accuracies
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {self.metadata_path}")
        
        # Plot history
        self._plot_history(history)
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        
        return model, history
    
    def _plot_history(self, history):
        """Plot training history with professional formatting"""
        # Separate plots for clarity and readability
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Accuracy plot
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(epochs, history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
        ax1.plot(epochs, history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
        ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('accuracy_plot_final.png', dpi=150, bbox_inches='tight')
        logger.info("Saved accuracy plot to accuracy_plot_final.png")
        plt.close()
        
        # Loss plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
        ax2.plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
        ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('loss_plot_final.png', dpi=150, bbox_inches='tight')
        logger.info("Saved loss plot to loss_plot_final.png")
        plt.close()


# ============================================================================
# CSV OUTPUT HANDLER - Store Prediction Results
# ============================================================================

class PredictionCSVWriter:
    """
    Handles writing pill prediction results to CSV files.
    Stores: pill name, confidence, usage, dosage, side effects, precautions
    """
    
    def __init__(self, output_dir='media/pilldata'):
        """
        Initialize CSV writer.
        
        Args:
            output_dir: Directory to store CSV files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.csv_file = os.path.join(output_dir, 'pill_predictions.csv')
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp',
                    'pill_name',
                    'confidence',
                    'usage',
                    'dosage',
                    'side_effects',
                    'precautions'
                ])
                writer.writeheader()
                logger.info(f"Created CSV file: {self.csv_file}")
    
    def write_prediction(self, pill_data):
        """
        Write a single prediction to CSV.
        
        Args:
            pill_data: Dictionary containing:
                - pill_name: str
                - confidence: str or float
                - usage: str
                - dosage: str
                - side_effects: str or list
                - precautions: str
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert lists to strings if needed
            side_effects = pill_data.get('side_effects', '')
            if isinstance(side_effects, list):
                side_effects = '; '.join(side_effects)
            
            row = {
                'timestamp': datetime.now().isoformat(),
                'pill_name': pill_data.get('pill_name', 'UNKNOWN'),
                'confidence': str(pill_data.get('confidence', '0%')),
                'usage': pill_data.get('usage', 'N/A'),
                'dosage': pill_data.get('dosage', 'N/A'),
                'side_effects': side_effects,
                'precautions': pill_data.get('precautions', 'Consult healthcare professional')
            }
            
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp',
                    'pill_name',
                    'confidence',
                    'usage',
                    'dosage',
                    'side_effects',
                    'precautions'
                ])
                writer.writerow(row)
            
            logger.info(f"Prediction recorded to CSV: {pill_data.get('pill_name', 'UNKNOWN')}")
            return True
        
        except Exception as e:
            logger.error(f"Error writing prediction to CSV: {str(e)}")
            return False
    
    def write_batch_predictions(self, pill_data_list):
        """
        Write multiple predictions to CSV.
        
        Args:
            pill_data_list: List of dictionaries containing pill data
        
        Returns:
            int: Number of successfully written predictions
        """
        count = 0
        for pill_data in pill_data_list:
            if self.write_prediction(pill_data):
                count += 1
        
        logger.info(f"Successfully wrote {count}/{len(pill_data_list)} predictions to CSV")
        return count
    
    def get_csv_path(self):
        """Get the path to the CSV file."""
        return self.csv_file
    
    def read_csv(self):
        """
        Read all predictions from CSV.
        
        Returns:
            pd.DataFrame: DataFrame with all predictions
        """
        if os.path.exists(self.csv_file):
            return pd.read_csv(self.csv_file)
        else:
            logger.warning(f"CSV file not found: {self.csv_file}")
            return pd.DataFrame()


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    trainer = PillClassifierTrainer(
        dataset_dir='media/train',
        model_output_path='media/model_feature_learning_mobilenetv3.keras'
    )
    
    model, history = trainer.train(
        epochs=20,
        batch_size=32,
        augmentation=True
    )
