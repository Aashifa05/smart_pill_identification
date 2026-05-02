#!/usr/bin/env python3
"""Minimal training script to diagnose issues"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import traceback

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    logger.info("Step 1: Importing TensorFlow...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    logger.info("  ✓ Imports successful")
    
    logger.info("\nStep 2: Loading data...")
    DATA_DIR = Path("media/pilldata")
    df_train = pd.read_csv(DATA_DIR / "Training_set.csv")
    df_test = pd.read_csv(DATA_DIR / "Testing_set.csv")
    logger.info(f"  ✓ Training: {len(df_train)}, Test: {len(df_test)} samples")
    
    logger.info("\nStep 3: Loading images...")
    X_train = []
    y_train = []
    for idx, row in df_train.iterrows():
        img_path = DATA_DIR / "train" / row['filename']
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        X_train.append(np.array(img, dtype=np.float32) / 255.0)
        y_train.append(row['label'])
        if (idx + 1) % 300 == 0:
            logger.info(f"  Loaded {idx + 1}/{len(df_train)}")
    
    X_train = np.array(X_train)
    X_test = []
    y_test = []
    for idx, row in df_test.iterrows():
        img_path = DATA_DIR / "train" / row['filename']
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        X_test.append(np.array(img, dtype=np.float32) / 255.0)
        y_test.append(row['label'])
    X_test = np.array(X_test)
    logger.info(f"  ✓ All images loaded: {X_train.shape}, {X_test.shape}")
    
    logger.info("\nStep 4: Encoding labels...")
    unique_labels = sorted(df_train['label'].unique())
    le = LabelEncoder()
    le.fit(unique_labels)
    y_train_enc = keras.utils.to_categorical(le.transform(y_train), len(unique_labels))
    y_test_enc = keras.utils.to_categorical(le.transform(y_test), len(unique_labels))
    logger.info(f"  ✓ Encoded: {len(unique_labels)} classes")
    
    logger.info("\nStep 5: Building model...")
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(unique_labels), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info(f"  ✓ Model built with {model.count_params():,} parameters")
    
    logger.info("\nStep 6: Training (5 epochs test)...")
    history = model.fit(X_train, y_train_enc, epochs=5, batch_size=32, validation_data=(X_test, y_test_enc), verbose=1)
    logger.info("  ✓ Training complete!")
    
    logger.info("\nStep 7: Evaluating...")
    loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    logger.info(f"  ✓ Test Accuracy: {acc*100:.2f}%")
    
    logger.info("\nStep 8: Saving model...")
    model_path = DATA_DIR / "model_imprint_robust_v2.h5"
    model.save(str(model_path))
    logger.info(f"  ✓ Model saved: {model_path.name} ({model_path.stat().st_size / (1024**2):.1f} MB)")
    
    metadata = {
        "classes": unique_labels,
        "num_classes": len(unique_labels),
        "test_accuracy": float(acc),
        "status": "successfully_trained"
    }
    with open(DATA_DIR / "model_imprint_robust_v2_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Metadata saved")
    
    logger.info("\n✓ SUCCESS! Model v2 trained and saved!")
    
except Exception as e:
    logger.error(f"\n✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
