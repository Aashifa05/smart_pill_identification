#!/usr/bin/env python
"""
Quick MobileNetV3 Training - Minimal dataset for FAST testing
Trains on 1000 images only to get a working model quickly
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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def get_data_paths():
    from django.conf import settings
    BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    return {
        'train_csv': BASE_DATA_PATH / 'Training_set.csv',
        'train_path': BASE_DATA_PATH / 'train',
        'model_path': BASE_DATA_PATH / 'model.h5',
        'model_metadata': BASE_DATA_PATH / 'model_metadata.json',
    }

def train_quick():
    print("\n" + "="*70)
    print("MOBILENETV3 QUICK TRAINING (1000 images - 5 epochs)")
    print("="*70 + "\n")
    
    paths = get_data_paths()
    
    # Load and limit data
    print("Loading data...")
    df = pd.read_csv(paths['train_csv'])
    
    # Take only 1000 samples (100 per class)
    labels = sorted(df['label'].unique())
    samples_per_class = 100
    df_limited = pd.concat([df[df['label'] == label].head(samples_per_class) for label in labels])
    
    print(f"  Using {len(df_limited)} images ({samples_per_class} per class)")
    
    label_map = {l: i for i, l in enumerate(labels)}
    reverse_label_map = {i: l for l, i in label_map.items()}
    df_limited['label_idx'] = df_limited['label'].map(label_map)
    
    # Split
    train_df, val_df = train_test_split(df_limited, test_size=0.2, random_state=42)
    
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}\n")
    
    # Data generators
    print("Setting up data pipeline...")
    datagen = ImageDataGenerator(rescale=1/127.5)
    
    train_gen = datagen.flow_from_dataframe(
        train_df,
        directory=paths['train_path'],
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    val_gen = datagen.flow_from_dataframe(
        val_df,
        directory=paths['train_path'],
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    print("Building MobileNetV3Large...")
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
    output = Dense(len(label_map), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    print(f"[OK] Model ready with {model.count_params():,} parameters\n")
    
    # Quick train (5 epochs)
    print("Training (5 epochs on 1000 images)...\n")
    history = model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
        verbose=1
    )
    
    # Save
    print("\nSaving model...")
    model.save(str(paths['model_path']))
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_shape': (224, 224, 3),
        'num_classes': len(label_map),
        'label_map': label_map,
        'reverse_label_map': {str(k): v for k, v in reverse_label_map.items()},
        'architecture': 'MobileNetV3Large + Transfer Learning',
        'note': 'Quick training on 1000 images - Run full training for better accuracy'
    }
    
    with open(paths['model_metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("\n" + "="*70)
    print("[OK] MODEL SAVED! Ready for predictions!")
    print("     Note: For better accuracy, run full training with all 7000 images")
    print("="*70 + "\n")

if __name__ == '__main__':
    train_quick()
