#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONFUSION MATRIX COLLAPSE DEBUG - PART 2
Focus on test set composition and actual confusion matrix generation
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
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.requirement import get_data_paths, preprocess_image_for_prediction

print("\n" + "=" * 100)
print("CONFUSION MATRIX DEBUG - PART 2: TEST SET ANALYSIS")
print("=" * 100)

# Load model
paths = get_data_paths()
model_path = paths['model_path']
metadata_path = paths['model_metadata']

print(f"\n📁 Loading model: {model_path.name}")
model = load_model(str(model_path))
print(f"✅ Model loaded: Input {model.input_shape} → Output {model.output_shape}")

# Load metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
label_map = metadata.get('label_map', {})
num_classes = len(label_map)

print(f"✅ Metadata loaded: {num_classes} classes")

# ============================================================================
# LOAD AND ANALYZE TEST SET
# ============================================================================
print("\n" + "-" * 100)
print("TEST SET ANALYSIS")
print("-" * 100)

# Load dataset from training directory
train_dir = paths['train_path']  # Use get_data_paths() result
print(f"\n📂 Loading dataset from: {train_dir}")

if not train_dir.exists():
    print(f"❌ Directory not found: {train_dir}")
    sys.exit(1)

images = []
labels_indices = []
class_counts = Counter()

for class_idx, (class_name, class_idx_from_map) in enumerate(sorted(label_map.items(), key=lambda x: x[1])):
    class_dir = train_dir / class_name
    
    if not class_dir.exists():
        print(f"⚠️  Class directory not found: {class_dir}")
        continue
    
    img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
    print(f"   {class_name}: {len(img_files)} images")
    
    for img_path in img_files:
        try:
            img_array = preprocess_image_for_prediction(str(img_path))
            if img_array.shape == (1, 224, 224, 3):
                images.append(img_array[0])  # Remove batch dimension
                labels_indices.append(class_idx_from_map)
                class_counts[class_name] += 1
        except Exception as e:
            print(f"      ⚠️  Error processing {img_path.name}: {e}")

print(f"\n✅ Loaded {len(images)} images from {len(class_counts)} classes")

if len(images) == 0:
    print("❌ No images loaded!")
    sys.exit(1)

images = np.array(images)
labels_indices = np.array(labels_indices)
labels_onehot = to_categorical(labels_indices, num_classes=num_classes)

print(f"\n📊 Dataset Statistics:")
print(f"   Total images: {len(images)}")
print(f"   Total classes: {num_classes}")
print(f"   Image shape: {images.shape}")
print(f"   Labels shape: {labels_onehot.shape}")

# Use stratified split to get test set (matching training)
from pill_project.prediction_admin.user_utility_files.datapipeline import DataBalancer

X_train, X_val, X_test, y_train, y_val, y_test = DataBalancer.stratified_split(
    images, labels_onehot
)

print(f"\n📊 Split Results:")
print(f"   Training:   {len(X_train):4d} images")
print(f"   Validation: {len(X_val):4d} images")
print(f"   Test:       {len(X_test):4d} images")

# ============================================================================
# ANALYZE TEST SET CLASS DISTRIBUTION
# ============================================================================
print("\n" + "-" * 100)
print("TEST SET CLASS DISTRIBUTION")
print("-" * 100)

y_test_indices = y_test.argmax(axis=1)
test_class_counts = Counter(y_test_indices)

print(f"\n📊 Test Set Class Distribution:")
for class_idx in sorted(test_class_counts.keys()):
    class_name = [k for k, v in label_map.items() if v == class_idx][0]
    count = test_class_counts[class_idx]
    percentage = (count / len(y_test_indices)) * 100
    print(f"   Class {class_idx:2d} ({class_name:30s}): {count:3d} images ({percentage:5.1f}%)")

print(f"\n🔍 Class Distribution Analysis:")
if max(test_class_counts.values()) / min(test_class_counts.values()) > 3:
    print(f"   ⚠️  WARNING: Significant class imbalance detected")
    print(f"      Max class size: {max(test_class_counts.values())}")
    print(f"      Min class size: {min(test_class_counts.values())}")
    print(f"      Ratio: {max(test_class_counts.values()) / min(test_class_counts.values()):.2f}x")
else:
    print(f"   ✅ Balanced distribution (max/min ratio: {max(test_class_counts.values()) / min(test_class_counts.values()):.2f}x)")

# ============================================================================
# GENERATE PREDICTIONS ON TEST SET
# ============================================================================
print("\n" + "-" * 100)
print("GENERATING PREDICTIONS ON TEST SET")
print("-" * 100)

print(f"\n🧠 Running predictions on {len(X_test)} test images...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_indices = y_pred_probs.argmax(axis=1)
y_test_indices = y_test.argmax(axis=1)

print(f"✅ Predictions generated")

# ============================================================================
# ANALYZE PREDICTION DISTRIBUTION
# ============================================================================
print("\n" + "-" * 100)
print("PREDICTION DISTRIBUTION ANALYSIS")
print("-" * 100)

pred_class_counts = Counter(y_pred_indices)
print(f"\n📊 Predicted Class Distribution:")
for class_idx in sorted(pred_class_counts.keys()):
    class_name = [k for k, v in label_map.items() if v == class_idx][0]
    count = pred_class_counts[class_idx]
    percentage = (count / len(y_pred_indices)) * 100
    print(f"   Class {class_idx:2d} ({class_name:30s}): {count:3d} predictions ({percentage:5.1f}%)")

print(f"\n🔍 Prediction Diversity Analysis:")
num_unique_predictions = len(pred_class_counts)
print(f"   Unique classes predicted: {num_unique_predictions} out of {num_classes}")

if num_unique_predictions < num_classes / 2:
    print(f"   ❌ PROBLEM: Model predicts less than half of classes")
else:
    print(f"   ✅ Model uses diverse predictions")

# Check for collapse
max_pred_class = max(pred_class_counts.values())
if max_pred_class > len(y_pred_indices) * 0.5:
    print(f"   ❌ COLLAPSE DETECTED: One class gets {max_pred_class/len(y_pred_indices)*100:.1f}% of predictions")
else:
    print(f"   ✅ No collapse")

# ============================================================================
# GENERATE CONFUSION MATRIX
# ============================================================================
print("\n" + "-" * 100)
print("CONFUSION MATRIX GENERATION")
print("-" * 100)

cm = confusion_matrix(y_test_indices, y_pred_indices)
print(f"\n✅ Confusion matrix generated: {cm.shape}")

# Analyze diagonal (correct predictions)
diagonal_sum = np.diag(cm).sum()
total_predictions = cm.sum()
accuracy = diagonal_sum / total_predictions

print(f"\n📊 Confusion Matrix Statistics:")
print(f"   Correct predictions (diagonal): {diagonal_sum}")
print(f"   Total predictions: {total_predictions}")
print(f"   Accuracy: {accuracy:.4f}")

# Analyze each class
print(f"\n📊 Per-Class Prediction Success:")
for class_idx in range(num_classes):
    class_name = [k for k, v in label_map.items() if v == class_idx][0]
    true_positives = cm[class_idx, class_idx]
    total_true = cm[class_idx, :].sum()
    
    if total_true > 0:
        recall = true_positives / total_true
    else:
        recall = 0
    
    total_pred = cm[:, class_idx].sum()
    if total_pred > 0:
        precision = true_positives / total_pred
    else:
        precision = 0
    
    if total_true > 0:
        print(f"   Class {class_idx:2d} ({class_name:30s}): TP={true_positives:3d}/{total_true:3d} (recall={recall:.2f})")

# ============================================================================
# CHECK FOR PATHOLOGICAL PATTERNS
# ============================================================================
print("\n" + "-" * 100)
print("PATHOLOGICAL PATTERN DETECTION")
print("-" * 100)

print(f"\n🔍 Checking for issues:")

# Check if model predicts same class for all
if len(pred_class_counts) == 1:
    most_predicted = list(pred_class_counts.keys())[0]
    most_predicted_name = [k for k, v in label_map.items() if v == most_predicted][0]
    print(f"   ❌ CRITICAL: All {len(y_pred_indices)} predictions are class {most_predicted} ({most_predicted_name})")
else:
    print(f"   ✅ Model predicts multiple classes")

# Check if model predicts different class than ground truth
wrong_predictions = (y_test_indices != y_pred_indices).sum()
wrong_percentage = (wrong_predictions / len(y_test_indices)) * 100
print(f"   Wrong predictions: {wrong_predictions}/{len(y_test_indices)} ({wrong_percentage:.1f}%)")

# Check if certain classes are always mispredicted
print(f"\n   Per-class misprediction rates:")
for class_idx in range(num_classes):
    class_name = [k for k, v in label_map.items() if v == class_idx][0]
    
    mask = y_test_indices == class_idx
    if mask.sum() > 0:
        total = mask.sum()
        wrong = (y_pred_indices[mask] != class_idx).sum()
        wrong_pct = (wrong / total) * 100
        
        # Show in compact format
        if wrong_pct > 50:
            status = "❌"
        elif wrong_pct > 20:
            status = "⚠️ "
        else:
            status = "✅"
        
        print(f"      {status} Class {class_idx:2d}: {wrong}/{total} wrong ({wrong_pct:.0f}%)")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"\n✅ Model Summary:")
print(f"   Total Parameters: {model.count_params():,}")
print(f"   Output Neurons: {model.output_shape[-1]}")
print(f"   Architecture: {model.summary()}")

print(f"\n📊 Test Set:")
print(f"   Total images: {len(X_test)}")
print(f"   Classes represented: {len(test_class_counts)}")

print(f"\n🎯 Predictions:")
print(f"   Unique classes predicted: {num_unique_predictions}/{num_classes}")
print(f"   Overall accuracy: {accuracy:.4f}")
print(f"   Collapse: {'❌ YES' if len(pred_class_counts) == 1 else '✅ NO'}")

print("\n" + "=" * 100 + "\n")
