#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast MobileNetV3 Training with Parallel Image Processing
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_data_paths():
    from django.conf import settings
    BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    return {
        'train_csv': BASE_DATA_PATH / 'Training_set.csv',
        'test_csv': BASE_DATA_PATH / 'Testing_set.csv',
        'train_path': BASE_DATA_PATH / 'train',
        'model_path': BASE_DATA_PATH / 'model.h5',
        'model_metadata': BASE_DATA_PATH / 'model_metadata.json',
    }

def load_image_quick(file_path):
    """Load and preprocess single image"""
    try:
        img = Image.open(file_path).convert('RGB')
        img.thumbnail((224, 224), Image.Resampling.LANCZOS)  # Fast thumbnail
        img_array = np.array(img, dtype='float32')
        # Pad if needed
        if img_array.shape[0] < 224 or img_array.shape[1] < 224:
            padded = np.zeros((224, 224, 3), dtype='float32')
            h, w = img_array.shape[:2]
            padded[:h, :w, :] = img_array
            img_array = padded
        img_array = (img_array / 127.5) - 1.0
        return img_array
    except:
        return None

def preprocess_parallel(filenames, base_path, max_workers=4):
    """Parallel image preprocessing"""
    images = []
    base_path = Path(base_path)
    total = len(filenames)
    processed = 0
    failed = 0
    
    print(f"Preprocessing {total} images (parallel, {max_workers} workers)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_image_quick, base_path / f): f for f in filenames}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                images.append(result)
                processed += 1
            else:
                failed += 1
            
            if (i + 1) % 500 == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                print(f"  {i + 1}/{total} ({pct:.0f}%) - Processed: {processed}, Failed: {failed}")
    
    print(f"[OK] Loaded {len(images)} images")
    return np.array(images)

def build_model(input_size, num_classes):
    """Build MobileNetV3 model"""
    print("Building MobileNetV3Large...")
    
    base_model = MobileNetV3Large(
        input_shape=input_size,
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax', name='pill_output')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    print(f"[OK] Model with {model.count_params():,} parameters")
    return model

def train():
    print("\n" + "="*70)
    print("PILL DETECTION - MobileNetV3 FAST TRAINING")
    print("="*70 + "\n")
    
    paths = get_data_paths()
    
    # Load data
    print("[1/5] Loading Data...")
    train_df = pd.read_csv(paths['train_csv'])
    test_df = pd.read_csv(paths['test_csv'])
    
    labels = sorted(set(train_df['label'].unique()) | set(test_df['label'].unique()))
    label_map = {l: i for i, l in enumerate(labels)}
    reverse_label_map = {i: l for l, i in label_map.items()}
    
    train_df['label_idx'] = train_df['label'].map(label_map)
    test_df['label_idx'] = test_df['label'].map(label_map)
    
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Classes: {labels}\n")
    
    # Preprocess
    print("[2/5] Preprocessing Images...")
    x_train_all = preprocess_parallel(train_df['filename'].values, paths['train_path'], max_workers=4)
    y_train_all = train_df['label_idx'].values[:len(x_train_all)]
    print()
    
    # Split
    print("[3/5] Splitting Data...")
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.19, random_state=42, stratify=y_temp
    )
    print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}\n")
    
    # Build
    print("[4/5] Building Model...")
    model = build_model((224, 224, 3), len(label_map))
    print()
    
    # Train
    print("[5/5] Training (Early stopping enabled)...\n")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        ],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=-1)
    acc = accuracy_score(y_test, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\n{'-'*70}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"{'-'*70}\n")
    
    # Save
    print("Saving model...")
    model.save(str(paths['model_path']))
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_shape': (224, 224, 3),
        'num_classes': len(label_map),
        'label_map': label_map,
        'reverse_label_map': {str(k): v for k, v in reverse_label_map.items()},
        'test_accuracy': float(acc),
        'test_precision': float(prec),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'architecture': 'MobileNetV3Large + Transfer Learning'
    }
    
    with open(paths['model_metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("[OK] TRAINING COMPLETE!")
    print("="*70 + "\n")

if __name__ == '__main__':
    train()
