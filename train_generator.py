#!/usr/bin/env python
"""
Fast MobileNetV3 Training - Using TensorFlow ImageDataGenerator for efficient loading
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_data_paths():
    from django.conf import settings
    BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    return {
        'train_csv': BASE_DATA_PATH / 'Training_set.csv',
        'train_path': BASE_DATA_PATH / 'train',
        'model_path': BASE_DATA_PATH / 'model.h5',
        'model_metadata': BASE_DATA_PATH / 'model_metadata.json',
    }

def train():
    print("\n" + "="*70)
    print("PILL DETECTION - MobileNetV3 TRAINING (TF Image Generator)")
    print("="*70 + "\n")
    
    paths = get_data_paths()
    
    # Load metadata
    print("[1/4] Loading Data...")
    train_df = pd.read_csv(paths['train_csv'])
    
    labels = sorted(train_df['label'].unique())
    label_map = {l: i for i, l in enumerate(labels)}
    reverse_label_map = {i: l for l, i in label_map.items()}
    
    train_df['label_idx'] = train_df['label'].map(label_map)
    
    print(f"  Total images: {len(train_df)}")
    print(f"  Classes ({len(labels)}): {labels}\n")
    
    # Create data generator
    print("[2/4] Setting up data pipeline...")
    
    datagen = ImageDataGenerator(
        rescale=1/127.5,  # Normalize to [-1, 1]
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # For validation/test
    val_datagen = ImageDataGenerator(rescale=1/127.5)
    
    # Split data
    train_df_temp, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['label']
    )
    train_df_final, test_df = train_test_split(
        train_df_temp, test_size=0.25, random_state=42, stratify=train_df_temp['label']
    )
    
    print(f"  Train: {len(train_df_final)}, Val: {len(val_df)}, Test: {len(test_df)}\n")
    
    # Build model
    print("[3/4] Building MobileNetV3Large...")
    base_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(len(label_map), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    print(f"  [OK] Model with {model.count_params():,} parameters\n")
    
    # Create generators
    train_generator = datagen.flow_from_dataframe(
        train_df_final,
        directory=paths['train_path'],
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        seed=42
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=paths['train_path'],
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        seed=42
    )
    
    # Train
    print("[4/4] Training (Early stopping enabled)...\n")
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        ],
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_generator = val_datagen.flow_from_dataframe(
        test_df,
        directory=paths['train_path'],
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        seed=42,
        shuffle=False
    )
    
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    
    print(f"\n{'-'*70}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
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
        'test_accuracy': float(test_acc),
        'architecture': 'MobileNetV3Large + Transfer Learning'
    }
    
    with open(paths['model_metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("[OK] MODEL SAVED!")
    print("="*70 + "\n")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
