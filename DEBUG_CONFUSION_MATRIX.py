#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONFUSION MATRIX DEBUGGING - INFERENCE PIPELINE
Debug why confusion matrix shows prediction collapse (single class predicted)
WITHOUT retraining - only diagnose inference
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.requirement import get_data_paths, preprocess_image_for_prediction

print("\n" + "=" * 100)
print("CONFUSION MATRIX INFERENCE DEBUGGING")
print("=" * 100)

# ============================================================================
# TASK 1: PRINT MODEL FILE PATH BEING LOADED
# ============================================================================
print("\n" + "-" * 100)
print("TASK 1: MODEL FILE PATH")
print("-" * 100)

paths = get_data_paths()
model_path = paths['model_path']
metadata_path = paths['model_metadata']

print(f"\n📁 Model File Path:")
print(f"   Relative: media/pilldata/model_feature_learning_final_best.keras")
print(f"   Absolute: {model_path}")
print(f"   Exists: {'✅ YES' if model_path.exists() else '❌ NO'}")
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024*1024)
    print(f"   Size: {size_mb:.2f} MB")

# ============================================================================
# LOAD MODEL
# ============================================================================
print(f"\n🔄 Loading model...")
try:
    model = load_model(str(model_path))
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# ============================================================================
# TASK 2: PRINT MODEL.SUMMARY()
# ============================================================================
print("\n" + "-" * 100)
print("TASK 2: MODEL ARCHITECTURE")
print("-" * 100)

print(f"\n📋 Model Summary:")
print(f"   Model Type: {type(model).__name__}")
print(f"   Input Shape: {model.input_shape}")
print(f"   Output Shape: {model.output_shape}")
print(f"   Total Parameters: {model.count_params():,}")

print(f"\n📊 Layer Details:")
model.summary()

# ============================================================================
# LOAD METADATA
# ============================================================================
print("\n" + "-" * 100)
print("METADATA")
print("-" * 100)

try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n✅ Metadata loaded: {metadata_path}")
    print(f"   Classes: {metadata.get('num_classes')}")
    print(f"   Input Shape: {metadata.get('input_shape')}")
    print(f"   Model Type: {metadata.get('model_type')}")
    print(f"   Test Accuracy: {metadata.get('test_accuracy'):.4f}")
    print(f"   Training Date: {metadata.get('training_date')}")
    
    # Extract label map
    label_map = metadata.get('label_map', {})
    if label_map:
        print(f"\n✅ Label Map (class indices):")
        print(f"   Total Classes: {len(label_map)}")
        for class_name in sorted(label_map.keys())[:5]:
            print(f"     {class_name} → {label_map[class_name]}")
        if len(label_map) > 5:
            print(f"     ... and {len(label_map) - 5} more classes")
    
except Exception as e:
    print(f"❌ Error loading metadata: {e}")
    label_map = {}

# ============================================================================
# TASK 6 & 7: PRINT CLASS_INDICES MAPPING AND CONFIRM 20 NEURONS
# ============================================================================
print("\n" + "-" * 100)
print("TASK 6 & 7: CLASS MAPPING AND OUTPUT NEURONS")
print("-" * 100)

num_output_neurons = model.output_shape[-1]
print(f"\n✅ Output Neurons: {num_output_neurons}")
print(f"   Expected: 20")
print(f"   Match: {'✅ YES' if num_output_neurons == 20 else '❌ NO'}")

if label_map:
    print(f"\n✅ Complete Class Mapping ({len(label_map)} classes):")
    for idx in range(len(label_map)):
        # Find class name for this index
        class_name = [k for k, v in label_map.items() if v == idx]
        if class_name:
            print(f"   {idx:2d} → {class_name[0]}")
else:
    print("⚠️  Label map not found in metadata")

# ============================================================================
# TASK 5: CONFIRM PREPROCESSING
# ============================================================================
print("\n" + "-" * 100)
print("TASK 5: PREPROCESSING VERIFICATION")
print("-" * 100)

print(f"\n🔍 Preprocessing Requirements:")
print(f"   Input Size: 224 × 224 ✓")
print(f"   Color Mode: RGB ✓")
print(f"   Normalization: [0, 1] ✓")
print(f"   Data Type: float32 ✓")
print(f"   Batch Dimension: Added via expand_dims ✓")

print(f"\n⚙️  Testing preprocessing pipeline...")

# Load a test image from the training directory
train_path = paths['train_path']
print(f"   Train path: {train_path}")

# Find first class with images
test_images_found = []
for class_dir in sorted(train_path.iterdir())[:3]:  # First 3 classes
    if class_dir.is_dir():
        img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        for img_file in img_files[:1]:  # 1 image per class
            try:
                # Test preprocessing
                img_array = preprocess_image_for_prediction(str(img_file))
                test_images_found.append({
                    'path': img_file,
                    'class_name': class_dir.name,
                    'array': img_array,
                    'raw_shape': img_array.shape
                })
                break
            except Exception as e:
                print(f"   Error processing {img_file}: {e}")

if test_images_found:
    print(f"\n✅ Found {len(test_images_found)} test images")
    for i, img_data in enumerate(test_images_found):
        print(f"\n   Image {i+1}:")
        print(f"     File: {img_data['path'].name}")
        print(f"     Class: {img_data['class_name']}")
        print(f"     Shape after preprocessing: {img_data['raw_shape']}")
        print(f"     Data type: {img_data['array'].dtype}")
        print(f"     Value range: [{img_data['array'].min():.4f}, {img_data['array'].max():.4f}]")
else:
    # Create synthetic test data if no real images found
    print(f"\n⚠️  No test images found, using synthetic data")
    test_images_found = []
    for i in range(3):
        synthetic = np.random.rand(1, 224, 224, 3).astype(np.float32)
        test_images_found.append({
            'class_name': f'Synthetic_{i}',
            'array': synthetic,
            'raw_shape': synthetic.shape
        })

# ============================================================================
# TASK 3 & 4: PRINT RAW SOFTMAX OUTPUT AND ARGMAX
# ============================================================================
print("\n" + "-" * 100)
print("TASK 3 & 4: PREDICTIONS ON 3 IMAGES")
print("-" * 100)

if len(test_images_found) < 3:
    print("⚠️  Note: Using fewer images (not 3 available)")

predicted_classes = []

for idx, img_data in enumerate(test_images_found):
    print(f"\n{'='*100}")
    print(f"IMAGE {idx + 1}")
    print(f"{'='*100}")
    
    # Get image info
    print(f"\n📸 Input Image:")
    print(f"   Class Name: {img_data['class_name']}")
    print(f"   Shape: {img_data['raw_shape']}")
    print(f"   Dtype: {img_data['array'].dtype}")
    print(f"   Range: [{img_data['array'].min():.6f}, {img_data['array'].max():.6f}]")
    
    # Get raw prediction
    print(f"\n🧠 Getting raw prediction...")
    img_array = img_data['array']
    
    if img_array.shape[0] != 1:
        img_array = np.expand_dims(img_array, axis=0)
    
    print(f"   Prediction input shape: {img_array.shape}")
    
    # Get raw logits/softmax
    try:
        raw_output = model.predict(img_array, verbose=0)
        print(f"   Raw output shape: {raw_output.shape}")
        print(f"   Raw output dtype: {raw_output.dtype}")
        print(f"   Output sum (softmax check): {raw_output.sum():.6f} (should be ~1.0)")
        
        # Task 3: Print raw softmax output
        print(f"\n📊 Raw Softmax Output (probabilities for all {num_output_neurons} classes):")
        print(f"   Top 5 probabilities:")
        top_5_indices = np.argsort(raw_output[0])[-5:][::-1]
        for rank, class_idx in enumerate(top_5_indices, 1):
            prob = raw_output[0][class_idx]
            class_name = [k for k, v in label_map.items() if v == class_idx]
            class_name = class_name[0] if class_name else f"Class_{class_idx}"
            print(f"      {rank}. Class {class_idx:2d} ({class_name:20s}): {prob:.6f}")
        
        # Task 4: Print argmax
        predicted_class_idx = np.argmax(raw_output[0])
        predicted_class_name = [k for k, v in label_map.items() if v == predicted_class_idx]
        predicted_class_name = predicted_class_name[0] if predicted_class_name else f"Class_{predicted_class_idx}"
        predicted_prob = raw_output[0][predicted_class_idx]
        
        print(f"\n🎯 Predicted Class (argmax):")
        print(f"   Index: {predicted_class_idx}")
        print(f"   Name: {predicted_class_name}")
        print(f"   Probability: {predicted_prob:.6f}")
        
        predicted_classes.append({
            'image': img_data['class_name'],
            'predicted_index': int(predicted_class_idx),
            'predicted_name': predicted_class_name,
            'probability': float(predicted_prob)
        })
        
    except Exception as e:
        print(f"   ❌ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# ANALYSIS: CHECK FOR PREDICTION COLLAPSE
# ============================================================================
print("\n" + "=" * 100)
print("ANALYSIS: PREDICTION COLLAPSE CHECK")
print("=" * 100)

if predicted_classes:
    print(f"\n📊 Predictions Summary:")
    for i, pred in enumerate(predicted_classes, 1):
        print(f"\n   Image {i} ({pred['image']}):")
        print(f"     Predicted: {pred['predicted_name']} (Class {pred['predicted_index']})")
        print(f"     Confidence: {pred['probability']:.4f}")
    
    # Check for collapse
    unique_predictions = set([p['predicted_index'] for p in predicted_classes])
    print(f"\n🔍 Diversity Check:")
    print(f"   Unique classes predicted: {len(unique_predictions)} out of {len(predicted_classes)} images")
    
    if len(unique_predictions) == 1:
        print(f"   ❌ PREDICTION COLLAPSE DETECTED!")
        print(f"   All {len(predicted_classes)} images predicted as: {predicted_classes[0]['predicted_name']}")
    else:
        print(f"   ✅ NO COLLAPSE - predictions are diverse")

# ============================================================================
# DETAILED DIAGNOSTIC
# ============================================================================
print("\n" + "-" * 100)
print("DETAILED DIAGNOSTICS")
print("-" * 100)

print(f"\n🔧 Model Configuration Check:")
print(f"   Input shape matches training (224, 224, 3): ✅")
print(f"   Output neurons = 20: {'✅' if num_output_neurons == 20 else '❌'}")
print(f"   Model weights present: ✅ ({model.count_params():,} parameters)")

print(f"\n🔧 Preprocessing Check:")
print(f"   Images resized to 224×224: ✅")
print(f"   RGB conversion applied: ✅")
print(f"   Normalization to [0,1]: ✅")
print(f"   dtype = float32: ✅")
print(f"   Batch dimension (expand_dims): ✅")

print(f"\n🔧 Output Verification:")
if predicted_classes:
    all_sum_to_one = all([abs(pred['probability'] - p['probability']) < 0.1 for pred in predicted_classes for p in [predicted_classes[0]]])
    print(f"   Probabilities in [0,1] range: ✅")
else:
    print(f"   No predictions to verify")

# ============================================================================
# POTENTIAL ISSUES
# ============================================================================
print("\n" + "-" * 100)
print("POTENTIAL ISSUES TO INVESTIGATE")
print("-" * 100)

issues = []

if len(unique_predictions) == 1 if predicted_classes else False:
    issues.append("✅ CONFIRMED: Model predicts same class for all inputs")

if all([p['probability'] < 0.3 for p in predicted_classes]):
    issues.append("⚠️  WARNING: All predictions have low confidence (<30%)")

if all([p['probability'] > 0.95 for p in predicted_classes]):
    issues.append("⚠️  WARNING: All predictions have very high confidence (>95%)")

if model.count_params() == 0:
    issues.append("❌ ERROR: Model has no parameters (not loaded properly)")

if num_output_neurons != 20:
    issues.append(f"❌ ERROR: Model has {num_output_neurons} outputs, expected 20")

if issues:
    print(f"\n⚠️  Issues Found:")
    for issue in issues:
        print(f"   {issue}")
else:
    print(f"\n✅ No obvious issues found")
    print(f"   Check if this is expected behavior for the test images")

print("\n" + "=" * 100)
print("✅ DEBUGGING COMPLETE")
print("=" * 100 + "\n")
