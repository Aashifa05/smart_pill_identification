#!/usr/bin/env python
"""
Fast MobileNetV3 Training - Skip intensive validation for faster training
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
from Users.utility.requirement import (
    get_data_paths, build_enhanced_model, create_callbacks, 
    create_data_augmentation, plot_training_history, evaluate_model,
    preprocess_image_for_prediction, predict_pill
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import json
import tensorflow as tf

def load_data_fast(train_csv_path, test_csv_path):
    """Load data without intensive validation"""
    print(f"Loading training data from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    
    print(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    
    # Create label maps
    unique_labels = sorted(set(train_df['label'].unique()) | set(test_df['label'].unique()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    
    # Convert labels to indices
    train_df['label_idx'] = train_df['label'].map(label_map)
    test_df['label_idx'] = test_df['label'].map(label_map)
    
    print(f"✓ Data loaded: {len(train_df)} training, {len(test_df)} test samples")
    print(f"✓ Classes: {list(unique_labels)}")
    
    return train_df, test_df, label_map, reverse_label_map

def preprocess_images_fast(filenames, base_path):
    """Fast preprocessing for MobileNetV3 (224x224 RGB)"""
    images = []
    base_path = Path(base_path)
    
    print(f"Preprocessing {len(filenames)} images (224x224 RGB)...")
    
    for idx, file in enumerate(filenames):
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(filenames)}")
        
        try:
            img = load_img(base_path / file, color_mode='rgb')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype='float32')
            # Normalize to [-1, 1]
            img_array = (img_array / 127.5) - 1.0
            images.append(img_array)
        except Exception as e:
            print(f"  Warning: {file} - {str(e)}")
            continue
    
    print(f"✓ Processed {len(images)} images successfully")
    return np.array(images)

def train_mobilenetv3():
    """Train MobileNetV3 model"""
    
    print("\n" + "="*70)
    print("PILL DETECTION - MobileNetV3 TRAINING")
    print("="*70)
    
    # Get paths
    paths = get_data_paths()
    train_csv = paths['train_csv']
    test_csv = paths['test_csv']
    train_img_path = paths['train_path']
    model_path = paths['model_path']
    
    # Load data
    print("\n1. Loading Data...")
    train_df, test_df, label_map, reverse_label_map = load_data_fast(train_csv, test_csv)
    
    # Preprocess images
    print("\n2. Preprocessing Images...")
    x_train_all = preprocess_images_fast(train_df['filename'], train_img_path)
    y_train_all = np.array(train_df['label_idx'].values, dtype=np.int32)
    
    # Split data: 70% train, 15% val, 15% test
    print("\n3. Splitting Data (70-15-15)...")
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.19, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    # Build model
    print("\n4. Building MobileNetV3Large Model...")
    input_size = (224, 224, 3)
    num_classes = len(label_map)
    model = build_enhanced_model(input_size, num_classes)
    print(f"✓ Model built with {model.count_params():,} parameters")
    
    # Create augmentation
    print("\n5. Setting Up Data Augmentation...")
    augmentation = create_data_augmentation()
    
    # Create callbacks
    print("\n6. Configuring Training...")
    callbacks = create_callbacks(str(model_path))
    
    # Train
    print("\n7. Starting Training (with Early Stopping)...\n")
    history = model.fit(
        augmentation.flow(x_train, y_train, batch_size=32),
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        steps_per_epoch=len(x_train) // 32,
        verbose=1
    )
    
    # Evaluate
    print("\n8. Evaluating Model...")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=-1)
    test_acc = accuracy_score(y_test, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")
    print(f"✓ Precision: {prec:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")
    
    # Save model
    print("\n9. Saving Model...")
    model.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
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
    
    metadata_path = paths['model_metadata']
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Metadata saved to {metadata_path}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"Your pill detection system is ready with MobileNetV3!")
    print("You can now upload images for prediction.\n")
    
    return test_acc, model, label_map, reverse_label_map

if __name__ == '__main__':
    try:
        train_mobilenetv3()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
