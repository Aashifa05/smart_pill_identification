#!/usr/bin/env python3
"""
Fast balanced training script for pill classification v2
Uses pre-trained MobileNetV3 with balanced augmentation
Works offline or with network issues
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= CONFIGURATION =============
DATA_DIR = Path("media/pilldata")
TRAIN_CSV = DATA_DIR / "Training_set.csv"
TEST_CSV = DATA_DIR / "Testing_set.csv"
TRAIN_IMG_DIR = DATA_DIR / "train"

MODEL_SAVE_PATH = DATA_DIR / "model_imprint_robust_v2.h5"
METADATA_SAVE_PATH = DATA_DIR / "model_imprint_robust_v2_metadata.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 5e-4

# ============= DATA LOADING =============
def load_images_efficient(csv_path, img_dir, max_images=None):
    """Load images efficiently with progress reporting"""
    df = pd.read_csv(csv_path)
    
    if max_images:
        df = df.head(max_images)
    
    images = []
    labels = []
    failed = 0
    
    for idx, row in df.iterrows():
        try:
            img_path = img_dir / row['filename']
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                failed += 1
                continue
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(row['label'])
            
            if (idx + 1) % 100 == 0:
                logger.info(f"  Loaded {idx + 1}/{len(df)} images")
        
        except Exception as e:
            logger.warning(f"Failed to load {row['filename']}: {e}")
            failed += 1
    
    logger.info(f"Successfully loaded {len(images)} images ({failed} failed)")
    return np.array(images), np.array(labels)

def build_model_offline(num_classes):
    """Build MobileNetV3 model with offline support"""
    logger.info("Building MobileNetV3 model...")
    
    try:
        # Try with weights - if it fails, build without weights
        try:
            base_model = MobileNetV3Large(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
        except Exception as e:
            logger.warning(f"Could not load ImageNet weights ({e}). Building without pre-trained weights.")
            base_model = MobileNetV3Large(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None
            )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build custom top layers
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model built successfully. Parameters: {model.count_params():,}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        raise

def main():
    logger.info("=" * 70)
    logger.info("PILL CLASSIFICATION v2 TRAINING (Balanced Augmentation)")
    logger.info("=" * 70)
    
    # 1. Load data
    logger.info("\nLoading data...")
    logger.info(f"Training CSV: {TRAIN_CSV}")
    
    # Read training data
    train_df = pd.read_csv(TRAIN_CSV)
    logger.info(f"Found {len(train_df)} training samples")
    
    # Get unique classes
    unique_labels = sorted(train_df['label'].unique())
    num_classes = len(unique_labels)
    logger.info(f"Found {num_classes} pill classes")
    logger.info(f"Classes: {unique_labels}")
    
    # Load images
    logger.info("\nLoading training images...")
    X_train, y_train_labels = load_images_efficient(TRAIN_CSV, TRAIN_IMG_DIR)
    
    logger.info("\nLoading test images...")
    X_test, y_test_labels = load_images_efficient(TEST_CSV, TRAIN_IMG_DIR)
    
    # Encode labels
    le = LabelEncoder()
    le.fit(unique_labels)
    y_train = keras.utils.to_categorical(
        le.transform(y_train_labels),
        num_classes=num_classes
    )
    y_test = keras.utils.to_categorical(
        le.transform(y_test_labels),
        num_classes=num_classes
    )
    
    logger.info(f"\nData shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_test: {y_test.shape}")
    
    # 2. Build model
    model = build_model_offline(num_classes)
    
    # 3. Setup augmentation
    logger.info("\nSetting up balanced augmentation...")
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False
    )
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        np.unique(np.argmax(y_train, axis=1)),
        np.argmax(y_train, axis=1)
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    logger.info(f"Class weights computed: {len(class_weight_dict)} classes")
    
    # 4. Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 5. Train model
    logger.info("\n" + "=" * 70)
    logger.info(f"Starting training for {EPOCHS} epochs")
    logger.info("=" * 70)
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 6. Evaluate
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 7. Save model
    logger.info("\n" + "=" * 70)
    logger.info("SAVING MODEL")
    logger.info("=" * 70)
    
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))
    logger.info(f"✓ Model saved: {MODEL_SAVE_PATH}")
    logger.info(f"  Size: {MODEL_SAVE_PATH.stat().st_size / (1024**2):.1f} MB")
    
    # 8. Save metadata
    metadata = {
        "classes": unique_labels,
        "num_classes": num_classes,
        "confidence_threshold": 0.5,
        "model_type": "MobileNetV3Large",
        "input_shape": [224, 224, 3],
        "augmentation": {
            "rotation_range": 15,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.1,
            "brightness_range": [0.9, 1.1]
        },
        "training": {
            "epochs": len(history.history['loss']),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "final_train_loss": float(history.history['loss'][-1]),
            "final_train_acc": float(history.history['accuracy'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_val_acc": float(history.history['val_accuracy'][-1]),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc)
        }
    }
    
    with open(METADATA_SAVE_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved: {METADATA_SAVE_PATH}")
    
    # 9. Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model v2 trained successfully!")
    logger.info(f"  Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"  Classes: {num_classes}")
    logger.info(f"  Model file: {MODEL_SAVE_PATH.name}")
    logger.info(f"  Ready for deployment!")
    logger.info("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
