#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robust MobileNetV3 Training with Error Handling and Progress Tracking
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

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import json

def get_data_paths():
    """Get portable data paths"""
    from django.conf import settings
    BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    return {
        'train_csv': BASE_DATA_PATH / 'Training_set.csv',
        'test_csv': BASE_DATA_PATH / 'Testing_set.csv',
        'train_path': BASE_DATA_PATH / 'train',
        'model_path': BASE_DATA_PATH / 'model.h5',
        'model_metadata': BASE_DATA_PATH / 'model_metadata.json',
    }

def load_data_fast(train_csv_path, test_csv_path):
    """Load data"""
    print(f"Loading training data from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    
    print(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    
    unique_labels = sorted(set(train_df['label'].unique()) | set(test_df['label'].unique()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    
    train_df['label_idx'] = train_df['label'].map(label_map)
    test_df['label_idx'] = test_df['label'].map(label_map)
    
    print(f"[OK] Data loaded: {len(train_df)} training, {len(test_df)} test samples")
    print(f"[OK] Classes: {list(unique_labels)}")
    
    return train_df, test_df, label_map, reverse_label_map

def load_and_preprocess_image(file_path):
    """Load and preprocess single image with robust error handling"""
    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype='float32')
        img_array = (img_array / 127.5) - 1.0
        return img_array
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {str(e)}")
        return None

def preprocess_images_robust(filenames, base_path, start_idx=0):
    """Preprocess images with progress tracking and recovery"""
    images = []
    base_path = Path(base_path)
    successful = 0
    failed = 0
    
    print(f"\nPreprocessing {len(filenames)} images...")
    
    for idx, file in enumerate(filenames):
        if (idx + 1) % 500 == 0 or (idx + 1) == len(filenames):
            pct = ((idx + 1) / len(filenames)) * 100
            print(f"  Progress: {idx + 1}/{len(filenames)} ({pct:.1f}%) - Successful: {successful}, Failed: {failed}")
        
        file_path = base_path / file
        img_array = load_and_preprocess_image(file_path)
        
        if img_array is not None:
            images.append(img_array)
            successful += 1
        else:
            failed += 1
    
    print(f"[OK] Processed {successful} images successfully ({failed} failed)")
    return np.array(images), successful

def build_mobilenetv3_model(input_size, num_classes):
    """Build MobileNetV3 model for transfer learning"""
    print("\nBuilding MobileNetV3Large model...")
    
    # Load pre-trained MobileNetV3
    base_model = MobileNetV3Large(
        input_shape=input_size,
        weights='imagenet',
        include_top=False
    )
    
    # Freeze base model
    base_model.trainable = False
    print(f"  Base model: {base_model.count_params():,} parameters (frozen)")
    
    # Custom head
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax', name='pill_output')(x)
    
    model = Model(inputs=base_model.input, outputs=output, name='PillMobileNetV3')
    
    # Compile
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    total_params = model.count_params()
    print(f"[OK] Model built: {total_params:,} total parameters")
    
    return model

def train_mobilenetv3():
    """Main training function"""
    
    print("\n" + "="*70)
    print("PILL DETECTION - MobileNetV3 RETRAINING")
    print("="*70)
    
    paths = get_data_paths()
    
    # Load data
    print("\n[1/6] Loading Data...")
    train_df, test_df, label_map, reverse_label_map = load_data_fast(
        paths['train_csv'], paths['test_csv']
    )
    
    # Preprocess images
    print("\n[2/6] Preprocessing Images...")
    x_train_all, train_success = preprocess_images_robust(train_df['filename'], paths['train_path'])
    y_train_all = np.array(train_df['label_idx'].values[:train_success], dtype=np.int32)
    
    if len(x_train_all) < 100:
        print("✗ Not enough valid images!")
        sys.exit(1)
    
    # Split data
    print("\n[3/6] Splitting Data (70-15-15)...")
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_train_all, y_train_all[:len(x_train_all)], test_size=0.2, 
        random_state=42, stratify=y_train_all[:len(x_train_all)]
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.19, random_state=42, stratify=y_temp
    )
    print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    # Build model
    print("\n[4/6] Building Model...")
    input_size = (224, 224, 3)
    num_classes = len(label_map)
    model = build_mobilenetv3_model(input_size, num_classes)
    
    # Create callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    
    # Train
    print("\n[5/6] Training Model...")
    print("  Note: Early stopping enabled - training will stop if validation loss plateaus\n")
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\n[6/6] Evaluating Model...")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=-1)
    test_acc = accuracy_score(y_test, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\n{'='*70}")
    print(f"Results on {len(x_test)} Test Images:")
    print(f"{'='*70}")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Save model
    print(f"\n{'='*70}")
    print("Saving Model...")
    model.save(str(paths['model_path']))
    print(f"[OK] Model saved to {paths['model_path']}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_shape': input_size,
        'num_classes': num_classes,
        'label_map': label_map,
        'reverse_label_map': {str(k): v for k, v in reverse_label_map.items()},
        'test_accuracy': float(test_acc),
        'test_precision': float(prec),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'training_samples': len(x_train),
        'validation_samples': len(x_val),
        'test_samples': len(x_test),
        'epochs_trained': len(history.history['loss']),
        'architecture': 'MobileNetV3Large with Transfer Learning'
    }
    
    with open(paths['model_metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Metadata saved")
    
    print(f"{'='*70}")
    print("[OK] TRAINING COMPLETE - Ready for predictions!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    try:
        train_mobilenetv3()
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
