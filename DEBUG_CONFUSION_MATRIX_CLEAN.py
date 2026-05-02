#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONFUSION MATRIX COLLAPSE DEBUG - SIMPLIFIED
Focus on actual test set and predictions from training code
Do NOT retrain, just run inf erence on test set
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.requirement import get_data_paths, preprocess_image_for_prediction

print("\n" + "=" * 100)
print("CONFUSION MATRIX COLLAPSE DEBUG - TEST SET INFERENCE")
print("=" * 100)

# Load model
paths = get_data_paths()
model_path = paths['model_path']
metadata_path = paths['model_metadata']

print(f"\n📁 Model File: {model_path.name}")
model = load_model(str(model_path))
print(f"✅ {model.input_shape} → {model.output_shape}")

# Load metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
label_map = metadata.get('label_map', {})
num_classes = len(label_map)

print(f"✅ {num_classes} classes loaded from metadata")

# ============================================================================
# LOAD TEST IMAGES MANUALLY
# ============================================================================
print("\n" + "-" * 100)
print("LOADING TEST IMAGES")
print("-" * 100)

train_dir = paths['train_path']
print(f"\n📂 Train directory: {train_dir}")
print(f"   Exists: {'✅ YES' if train_dir.exists() else '❌ NO'}")

if not train_dir.exists():
    print(f"❌ Cannot access training directory")
    sys.exit(1)

# List subdirectories
subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
print(f"\n   Found {len(subdirs)} class directories")

images = []
labels_indices = []
class_counts = Counter()

for class_dir in sorted(subdirs)[:20]:  # Limit to 20 classes
    class_name = class_dir.name
    if class_name not in label_map:
        print(f"   ⚠️  {class_name} not in label map, skipping")
        continue
    
    class_idx = label_map[class_name]
    
    img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
    print(f"   {class_name:35s}: {len(img_files):3d} images", end='')
    
    count_loaded = 0
    for img_path in img_files:
        try:
            img_array = preprocess_image_for_prediction(str(img_path))
            if img_array is not None and img_array.shape == (1, 224, 224, 3):
                images.append(img_array[0])  # Remove batch dim
                labels_indices.append(class_idx)
                count_loaded += 1
                class_counts[class_name] += 1
        except:
            pass
    
    if count_loaded > 0:
        print(f" ✅ {count_loaded} loaded")
    else:
        print(f" ❌")

print(f"\n✅ Total: {len(images)} images from {len(class_counts)} classes")

if len(images) == 0:
    print("❌ No images loaded!")
    sys.exit(1)

images = np.array(images)
labels_indices = np.array(labels_indices)
labels_onehot = to_categorical(labels_indices, num_classes=num_classes)

print(f"   Shape: {images.shape}")
print(f"   Labels shape: {labels_onehot.shape}")

# ============================================================================
# USE STRATIFIED SPLIT (SAME AS TRAINING)
# ============================================================================
print("\n" + "-" * 100)
print("SPLITTING DATA (stratified)")
print("-" * 100)

from pill_project.prediction_admin.user_utility_files.datapipeline import DataBalancer

X_train, X_val, X_test, y_train, y_val, y_test = DataBalancer.stratified_split(
    images, labels_onehot
)

print(f"\n✅ Train: {len(X_train):4d} | Val: {len(X_val):4d} | Test: {len(X_test):4d}")

# ============================================================================
# ANALYZE TEST SET
# ============================================================================
print("\n" + "-" * 100)
print("TEST SET DETAILS")
print("-" * 100)

y_test_indices = y_test.argmax(axis=1)
test_counts = Counter(y_test_indices)

print(f"\n✅ Test set class distribution:")
for idx in sorted(test_counts.keys()):
    class_name = [k for k, v in label_map.items() if v == idx][0]
    count = test_counts[idx]
    print(f"   Class {idx:2d} ({class_name:30s}): {count:3d} images")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================
print("\n" + "-" * 100)
print("MAKING PREDICTIONS")
print("-" * 100)

print(f"\n🧠 Predicting on {len(X_test)} test images...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_indices = np.argmax(y_pred_probs, axis=1)

print(f"✅ Predictions complete")

# ============================================================================
# ANALYZE PREDICTIONS
# ============================================================================
print("\n" + "-" * 100)
print("PREDICTION ANALYSIS")
print("-" * 100)

pred_counts = Counter(y_pred_indices)
print(f"\n📊 Predicted class distribution:")
for idx in sorted(pred_counts.keys()):
    class_name = [k for k, v in label_map.items() if v == idx][0]
    count = pred_counts[idx]
    pct = (count / len(y_pred_indices)) * 100
    print(f"   Class {idx:2d} ({class_name:30s}): {count:3d} ({pct:5.1f}%)")

print(f"\n🔍 Prediction Statistics:")
print(f"   Unique classes predicted: {len(pred_counts)} / {num_classes}")

if len(pred_counts) == 1:
    most_pred = list(pred_counts.keys())[0]
    most_pred_name = [k for k, v in label_map.items() if v == most_pred][0]
    print(f"   ❌ COLLAPSE: All {len(y_pred_indices)} predictions are class {most_pred} ({most_pred_name})")
else:
    print(f"   ✅ Diverse predictions")

# ============================================================================
# COMPUTE CONFUSION MATRIX
# ============================================================================
print("\n" + "-" * 100)
print("CONFUSION MATRIX")
print("-" * 100)

cm = confusion_matrix(y_test_indices, y_pred_indices)
accuracy = accuracy_score(y_test_indices, y_pred_indices)

print(f"\n✅ Confusion matrix: {cm.shape}")
print(f"   Accuracy: {accuracy:.4f}")

# Per-class metrics
class_names_sorted = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

report = classification_report(
    y_test_indices, y_pred_indices,
    target_names=class_names_sorted,
    output_dict=False
)

print(f"\n📊 Classification Report:")
print(report)

# ============================================================================
# DIAGNOSTIC
# ============================================================================
print("\n" + "=" * 100)
print("DIAGNOSTIC SUMMARY")
print("=" * 100)

print(f"\n✅ Model Details:")
print(f"   Input:  {model.input_shape}")
print(f"   Output: {model.output_shape}")
print(f"   Params: {model.count_params():,}")

print(f"\n✅ Test Set:")
print(f"   Images: {len(X_test)}")
print(f"   Classes: {len(test_counts)}")
print(f"   Largest class: {max(test_counts.values())} images")
print(f"   Smallest class: {min(test_counts.values())} images")

print(f"\n✅ Predictions:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Unique classes: {len(pred_counts)} / {num_classes}")
print(f"   Collapse: {'❌ YES' if len(pred_counts) == 1 else '✅ NO'}")

# Check specific patterns
print(f"\n🔍 Looking for issues:")
if len(pred_counts) < 5:
    print(f"   ⚠️  Model predicts <5 unique classes (collapse or severe bias)")
elif len(pred_counts) < num_classes / 2:
    print(f"   ⚠️  Model predicts <50% of classes")
else:
    print(f"   ✅ Reasonable prediction diversity")

if accuracy < 0.3:
    print(f"   ⚠️  Very low accuracy ({accuracy:.2%})")
elif accuracy > 0.9:
    print(f"   ✅ High accuracy ({accuracy:.2%})")
else:
    print(f"   ✅ Reasonable accuracy ({accuracy:.2%})")

print("\n" + "=" * 100 + "\n")
