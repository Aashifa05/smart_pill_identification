#!/usr/bin/env python3
"""
Fast training with custom CNN (no pre-trained weights needed)
Uses balanced augmentation to fix UNKNOWN TABLET predictions
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
EPOCHS = 40
LEARNING_RATE = 1e-3

# ============= DATA LOADING =============
def load_images(csv_path, img_dir):
    """Load all images into memory"""
    df = pd.read_csv(csv_path)
    
    images = []
    labels = []
    failed = 0
    
    for idx, row in df.iterrows():
        try:
            img_path = img_dir / row['filename']
            if not img_path.exists():
                failed += 1
                continue
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(row['label'])
            
            if (idx + 1) % 200 == 0:
                logger.info(f"  Loaded {idx + 1}/{len(df)} images")
        
        except Exception as e:
            failed += 1
    
    logger.info(f"Successfully loaded {len(images)} images ({failed} failed)")
    return np.array(images), np.array(labels)

def build_custom_cnn(num_classes):
    """Build lightweight custom CNN"""
    logger.info("Building custom CNN model...")
    
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built. Parameters: {model.count_params():,}")
    return model

def main():
    logger.info("=" * 70)
    logger.info("PILL CLASSIFICATION v2 - CUSTOM CNN TRAINING")
    logger.info("=" * 70)
    
    # 1. Load data
    logger.info("\nLoading training data...")
    train_df = pd.read_csv(TRAIN_CSV)
    logger.info(f"Found {len(train_df)} training samples")
    
    unique_labels = sorted(train_df['label'].unique())
    num_classes = len(unique_labels)
    logger.info(f"Found {num_classes} pill classes")
    
    logger.info("\nLoading images...")
    X_train, y_train_labels = load_images(TRAIN_CSV, TRAIN_IMG_DIR)
    X_test, y_test_labels = load_images(TEST_CSV, TRAIN_IMG_DIR)
    
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
    
    logger.info(f"Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # 2. Build model
    model = build_custom_cnn(num_classes)
    
    # 3. Setup augmentation (BALANCED - not aggressive)
    logger.info("\nSetting up balanced augmentation...")
    train_datagen = ImageDataGenerator(
        rotation_range=15,           # ±15° rotation
        width_shift_range=0.1,       # ±10% shift
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,              # ±10% zoom
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
    logger.info(f"Class weights computed for {len(class_weight_dict)} classes")
    
    # 4. Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    # 5. Train
    logger.info("\n" + "=" * 70)
    logger.info(f"Training for {EPOCHS} epochs (with early stopping)")
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
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info("=" * 70)
    
    # 7. Save model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))
    logger.info(f"✓ Model saved to {MODEL_SAVE_PATH.name}")
    logger.info(f"  Size: {MODEL_SAVE_PATH.stat().st_size / (1024**2):.1f} MB")
    
    # 8. Save metadata
    metadata = {
        "classes": unique_labels,
        "num_classes": num_classes,
        "confidence_threshold": 0.5,
        "model_type": "Custom CNN",
        "input_shape": [224, 224, 3],
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "augmentation": "balanced (rotation 15°, shift 10%, zoom 10%)"
    }
    
    with open(METADATA_SAVE_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved")
    
    # 9. Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - Model v2 Ready!")
    logger.info("=" * 70)
    logger.info(f"✓ Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"✓ Classes: {num_classes} pill types")
    logger.info(f"✓ Model file: {MODEL_SAVE_PATH.name}")
    logger.info(f"✓ Ready for deployment in Django!")
    logger.info("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
