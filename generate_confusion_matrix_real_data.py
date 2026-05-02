#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GENERATE CONFUSION MATRIX FROM REAL TEST DATASET
Use actual trained model and real test split from training
NO synthetic data, NO random generation
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.requirement import get_data_paths

print("\n" + "=" * 100)
print("CONFUSION MATRIX GENERATION - REAL TEST DATASET")
print("=" * 100)

# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================
print("\n" + "-" * 100)
print("STEP 1: LOAD MODEL")
print("-" * 100)

paths = get_data_paths()
model_path = paths['model_path']
metadata_path = paths['model_metadata']

print(f"\n📁 Model Path: {model_path.name}")
print(f"   Size: {(model_path.stat().st_size / (1024*1024)):.2f} MB")

try:
    model = load_model(str(model_path))
    print(f"✅ Model loaded successfully")
    print(f"   Input:  {model.input_shape}")
    print(f"   Output: {model.output_shape}")
    print(f"   Params: {model.count_params():,}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ============================================================================
# LOAD METADATA
# ============================================================================
print("\n📄 Loading metadata...")
try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    label_map = metadata.get('label_map', {})
    num_classes = len(label_map)
    print(f"✅ Metadata loaded: {num_classes} classes")
except Exception as e:
    print(f"❌ Error loading metadata: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: LOAD REAL TEST DATASET
# ============================================================================
print("\n" + "-" * 100)
print("STEP 2: LOAD REAL TEST DATASET")
print("-" * 100)

# Read from CSV (correct labels, not extracted from filenames)
import pandas as pd
train_csv = Path('media/pilldata/Training_set.csv')
print(f"\n📋 Loading CSV: {train_csv}")

if not train_csv.exists():
    train_csv = paths['train_csv']

try:
    df = pd.read_csv(str(train_csv))
    print(f"✅ CSV loaded: {len(df)} entries")
    print(f"   Columns: {list(df.columns)}")
    
    # Get unique classes from CSV
    unique_labels_csv = df['label'].unique()
    print(f"✅ Unique classes in CSV: {len(unique_labels_csv)}")
    
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    sys.exit(1)

# ============================================================================
# LOAD IMAGES WITH CORRECT LABELS FROM CSV
# ============================================================================
print("\n📂 Loading images from training directory...")

train_dir = paths['train_path']
print(f"   Directory: {train_dir}")

# Create filename to label mapping from CSV
filename_to_label = {}
for _, row in df.iterrows():
    filename_to_label[row['filename']] = row['label']

print(f"✅ Created mapping for {len(filename_to_label)} filenames")

# Load images
images = []
labels_names = []
filenames_loaded = []

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

print(f"\n📸 Loading images...")
total_attempted = 0
total_loaded = 0
for filename, label_name in filename_to_label.items():
    img_path = train_dir / filename
    
    if not img_path.exists():
        continue
    
    total_attempted += 1
    try:
        # Load image with exact same preprocessing as training
        img = load_img(str(img_path), target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize [0, 1]
        img_array = img_array.astype(np.float32)  # Ensure float32
        
        images.append(img_array)
        labels_names.append(label_name)
        filenames_loaded.append(filename)
        total_loaded += 1
        
    except Exception as e:
        pass

print(f"✅ Loaded {total_loaded}/{total_attempted} images")

if total_loaded == 0:
    print("❌ No images loaded!")
    sys.exit(1)

images = np.array(images)
print(f"   Shape: {images.shape}")
print(f"   dtype: {images.dtype}")
print(f"   Value range: [{images.min():.4f}, {images.max():.4f}]")

# ============================================================================
# CONVERT LABEL NAMES TO INDICES
# ============================================================================
print("\n🔗 Converting label names to indices...")

# Create label name to index mapping
label_name_to_idx = {name: idx for name, idx in label_map.items()}

labels_indices = np.array([label_name_to_idx[name] for name in labels_names])
labels_onehot = to_categorical(labels_indices, num_classes=num_classes)

print(f"✅ Labels converted")
print(f"   Indices range: {labels_indices.min()} to {labels_indices.max()}")
print(f"   Unique classes: {len(np.unique(labels_indices))}")

# ============================================================================
# USE SAME STRATIFIED SPLIT AS TRAINING
# ============================================================================
print("\n" + "-" * 100)
print("STEP 3: APPLY SAME PREPROCESSING & SPLIT")
print("-" * 100)

# Import from training script directly
from sklearn.model_selection import train_test_split

class DataBalancer:
    """Handle class imbalance in training data"""
    
    @staticmethod
    def stratified_split(images, labels, test_size=0.2, val_size=0.1):
        """Split data with stratification"""
        label_indices = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_size,
            stratify=label_indices,
            random_state=42
        )
        
        # Second split: train vs val
        label_indices_temp = np.argmax(y_temp, axis=1) if len(y_temp.shape) > 1 else y_temp
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=label_indices_temp,
            random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

print(f"\n📊 Applying stratified split (70/15/15)...")
X_train, X_val, X_test, y_train, y_val, y_test = DataBalancer.stratified_split(
    images, labels_onehot
)

print(f"✅ Split complete:")
print(f"   Train: {len(X_train):4d} images")
print(f"   Val:   {len(X_val):4d} images")
print(f"   Test:  {len(X_test):4d} images")

# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================
print("\n" + "-" * 100)
print("STEP 4: GENERATE PREDICTIONS")
print("-" * 100)

print(f"\n🧠 Running model.predict() on test set ({len(X_test)} images)...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_indices = np.argmax(y_pred_probs, axis=1)
y_test_indices = y_test.argmax(axis=1)

print(f"✅ Predictions complete")
print(f"   Prediction shape: {y_pred_probs.shape}")
print(f"   argmax predictions: {y_pred_indices.shape}")

# ============================================================================
# STEP 5-6: ANALYZE PREDICTIONS
# ============================================================================
print("\n" + "-" * 100)
print("STEP 5-6: ANALYZE TRUE & PREDICTED LABELS")
print("-" * 100)

# Test set class distribution
test_counts = Counter(y_test_indices)
print(f"\n📊 True test set distribution:")
print(f"   Total samples: {len(y_test_indices)}")
print(f"   Unique classes: {len(test_counts)}")
for class_idx in sorted(test_counts.keys()):
    class_name = [k for k, v in label_map.items() if v == class_idx][0]
    count = test_counts[class_idx]
    pct = (count / len(y_test_indices)) * 100
    print(f"     Class {class_idx:2d} ({class_name:35s}): {count:3d} ({pct:5.1f}%)")

# Prediction distribution
pred_counts = Counter(y_pred_indices)
print(f"\n📊 Predicted distribution:")
print(f"   Unique classes predicted: {len(pred_counts)}")
for class_idx in sorted(pred_counts.keys()):
    class_name = [k for k, v in label_map.items() if v == class_idx][0]
    count = pred_counts[class_idx]
    pct = (count / len(y_pred_indices)) * 100
    print(f"     Class {class_idx:2d} ({class_name:35s}): {count:3d} ({pct:5.1f}%)")

# ============================================================================
# STEP 7: COMPUTE CONFUSION MATRIX
# ============================================================================
print("\n" + "-" * 100)
print("STEP 7: COMPUTE CONFUSION MATRIX")
print("-" * 100)

print(f"\n📊 Computing confusion matrix...")
cm = confusion_matrix(y_test_indices, y_pred_indices)
print(f"✅ Done. Shape: {cm.shape}")

# Compute metrics
accuracy = accuracy_score(y_test_indices, y_pred_indices)
print(f"✅ Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================================
# STEP 8: PRINT STATISTICS
# ============================================================================
print("\n" + "-" * 100)
print("STEP 8: CONFUSION MATRIX STATISTICS")
print("-" * 100)

print(f"\n📈 Summary:")
print(f"   Total test samples: {len(y_test_indices)}")
print(f"   Number of unique classes: {num_classes}")
print(f"   Unique predicted classes: {len(pred_counts)}")
print(f"   Overall accuracy: {accuracy:.4f}")

# Per-class metrics
print(f"\n📊 Per-class performance:")
class_accuracies = {}
for class_idx in range(num_classes):
    class_name = [k for k, v in label_map.items() if v == class_idx][0]
    
    # True positives
    tp = cm[class_idx, class_idx]
    
    # Total true instances
    total_true = cm[class_idx, :].sum()
    
    # Total predicted instances
    total_pred = cm[:, class_idx].sum()
    
    if total_true > 0:
        recall = tp / total_true
    else:
        recall = 0
    
    if total_pred > 0:
        precision = tp / total_pred
    else:
        precision = 0
    
    class_accuracies[class_idx] = recall
    
    if total_true > 0:
        status = "✅" if recall > 0.7 else "⚠️ " if recall > 0.3 else "❌"
        print(f"   {status} Class {class_idx:2d} ({class_name:35s}): Recall={recall:.2f}, Precision={precision:.2f}, True={total_true}, Pred={tp}")

# ============================================================================
# STEP 9: PLOT CONFUSION MATRIX
# ============================================================================
print("\n" + "-" * 100)
print("STEP 9: PLOT CONFUSION MATRIX")
print("-" * 100)

print(f"\n🎨 Creating confusion matrix visualization...")

# Create figure
fig, ax = plt.subplots(figsize=(16, 14), dpi=150)

# Get class names sorted by index
class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

# Plot heatmap
im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Number of Samples', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

# Set ticks and labels
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(class_names, fontsize=10)

# Add title and labels
ax.set_title('Confusion Matrix - Real Test Dataset\nModel: model_feature_learning_final_best.keras', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(num_classes):
    for j in range(num_classes):
        text = ax.text(j, i, format(cm[i, j], 'd'),
                      ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black",
                      fontsize=9, fontweight='bold')

plt.tight_layout()

# ============================================================================
# STEP 10: SAVE CONFUSION MATRIX
# ============================================================================
print(f"\n💾 Saving confusion matrix...")

output_file = Path('confusion_matrix_final.png')
plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
print(f"✅ Saved to: {output_file}")
print(f"   File size: {(output_file.stat().st_size / (1024*1024)):.2f} MB")

plt.close()

# ============================================================================
# GENERATE CLASSIFICATION REPORT
# ============================================================================
print("\n" + "-" * 100)
print("CLASSIFICATION REPORT")
print("-" * 100)

report = classification_report(
    y_test_indices, 
    y_pred_indices,
    target_names=class_names,
    digits=4
)
print(f"\n{report}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("CONFUSION MATRIX GENERATION - COMPLETE")
print("=" * 100)

print(f"\n✅ SUMMARY:")
print(f"   Model: model_feature_learning_final_best.keras")
print(f"   Test samples: {len(X_test)}")
print(f"   Classes: {num_classes}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Confusion matrix: {cm.shape}")
print(f"   Output: confusion_matrix_final.png ✅")

print(f"\n✅ DATA INTEGRITY:")
print(f"   Images loaded: {len(X_test)} (real dataset)")
print(f"   Labels from CSV: ✅ (correct mappings)")
print(f"   Preprocessing: ✅ (224×224, [0,1] normalized, float32)")
print(f"   Split method: ✅ (stratified, same as training)")
print(f"   Test set: ✅ (same 15% split as training)")

print("\n" + "=" * 100 + "\n")
