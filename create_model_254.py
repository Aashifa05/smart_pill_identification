#!/usr/bin/env python
"""
Create a new MobileNetV3 model for 254 pill classes from Kaggle dataset
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import json
from datetime import datetime
import numpy as np

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.optimizers import Adam

def get_data_paths():
    from django.conf import settings
    BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    return {
        'model_path': BASE_DATA_PATH / 'model.keras',
        'model_metadata': BASE_DATA_PATH / 'model_metadata.json',
    }

def create_model_for_254_classes():
    """Create and save a MobileNetV3 model for 254 pill classes"""
    print("\n🚀 Creating MobileNetV3 Model for 254 Pill Classes...")
    print("="*70)

    paths = get_data_paths()

    # 254 pill classes from Kaggle dataset
    num_classes = 254

    print(f"\n[1] Building MobileNetV3Large base model...")
    base_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False
    print(f"    Base model: {base_model.count_params():,} parameters")

    print(f"\n[2] Adding classification layers for {num_classes} classes...")
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    print(f"    Classification head: ~70K parameters")
    print(f"    Total model: {model.count_params():,} parameters")

    print(f"\n[3] Saving model to {paths['model_path']}...")
    model.save(str(paths['model_path']))

    # Create label mapping for 254 classes (0-253)
    label_map = {i: i for i in range(num_classes)}
    reverse_label_map = {i: i for i in range(num_classes)}

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_shape': [224, 224, 3],
        'num_classes': num_classes,
        'label_map': label_map,
        'reverse_label_map': {str(k): v for k, v in reverse_label_map.items()},
        'test_accuracy': 0.0,
        'test_precision': 0.0,
        'test_recall': 0.0,
        'test_f1': 0.0,
        'architecture': 'MobileNetV3Large + Transfer Learning (254 classes)',
        'status': 'Model structure created - Ready for training or use',
        'dataset': 'Kaggle 1K Pill Image Dataset',
        'training_images': 481,
        'testing_images': 253
    }

    with open(paths['model_metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"    Metadata saved")

    print("\n" + "="*70)
    print("[OK] MODEL READY FOR 254 CLASSES!")
    print("="*70)
    print(f"\nYour system is now configured for:")
    print(f"  - 254 different pill types")
    print(f"  - 481 training images")
    print(f"  - 253 testing images")
    print(f"  - MobileNetV3Large architecture")
    print(f"  - Model location: {paths['model_path']}\n")

if __name__ == '__main__':
    try:
        create_model_for_254_classes()
    except Exception as e:
        print(f"\n[ERROR] Failed to create model: {e}")
        import traceback
        traceback.print_exc()