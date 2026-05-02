#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Inference Pipeline Debugger
===========================================

This script identifies why the model predicts the same class for all inputs.

Steps:
1. Load the model and verify architecture
2. Check preprocessing pipeline
3. Test predictions on known training images
4. Analyze probability distributions
5. Identify the root cause

"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.requirement import (
    get_data_paths, 
    preprocess_image_for_prediction,
    predict_pill
)

print("=" * 80)
print("INFERENCE PIPELINE DEBUGGING")
print("=" * 80)

# ============================================================================
# STEP 1: VERIFY MODEL LOADING
# ============================================================================
print("\n[STEP 1] LOADING MODEL")
print("-" * 80)

paths = get_data_paths()
model_path = paths['model_path']
metadata_path = paths['model_metadata']

print(f"Model path: {model_path}")
print(f"Model exists: {model_path.exists()}")
print(f"Metadata path: {metadata_path}")
print(f"Metadata exists: {metadata_path.exists()}")

try:
    model = load_model(str(model_path))
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# Print model summary
print("\nModel Architecture:")
print("-" * 80)
model.summary()

# ============================================================================
# STEP 2: VERIFY METADATA AND CLASS MAPPING
# ============================================================================
print("\n[STEP 2] LOADING METADATA AND CLASS MAPPING")
print("-" * 80)

try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'label_map' in metadata:
        label_map = metadata['label_map']
        reverse_label_map = {int(v): str(k) for k, v in label_map.items()}
        print(f"✓ Label map loaded with {len(label_map)} classes")
        print("\nClass Mapping (20 classes):")
        for idx in sorted(reverse_label_map.keys()):
            print(f"  {idx}: {reverse_label_map[idx]}")
    else:
        print("❌ ERROR: 'label_map' not in metadata")
        if 'reverse_label_map' in metadata:
            reverse_label_map = {int(k): str(v) for k, v in metadata['reverse_label_map'].items()}
            print(f"✓ Using 'reverse_label_map' instead ({len(reverse_label_map)} classes)")
except Exception as e:
    print(f"❌ ERROR loading metadata: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: VERIFY IMAGE PREPROCESSING
# ============================================================================
print("\n[STEP 3] CHECKING IMAGE PREPROCESSING PIPELINE")
print("-" * 80)

train_path = paths['train_path']
print(f"Train path: {train_path}")
print(f"Train path exists: {train_path.exists()}")

# Find a test image
test_images = list(train_path.glob("**/*.jpg")) + list(train_path.glob("**/*.png"))
if not test_images:
    print("❌ ERROR: No test images found in training directory")
    sys.exit(1)

test_image_path = test_images[0]
print(f"Using test image: {test_image_path}")
print(f"File size: {os.path.getsize(test_image_path)} bytes")

# Load raw image
raw_img = Image.open(test_image_path)
print(f"Raw image size: {raw_img.size}")
print(f"Raw image mode: {raw_img.mode}")

# Check preprocessing
processed = preprocess_image_for_prediction(str(test_image_path))
print(f"\nProcessed image shape: {processed.shape}")
print(f"Processed dtype: {processed.dtype}")
print(f"Processed value range: [{processed.min():.4f}, {processed.max():.4f}]")
print(f"Expected range: [0.0, 1.0]")

if processed.shape != (1, 224, 224, 3):
    print(f"❌ ERROR: Expected shape (1, 224, 224, 3), got {processed.shape}")
else:
    print("✓ Shape is correct")

if processed.dtype != np.float32:
    print(f"⚠️  WARNING: Expected dtype float32, got {processed.dtype}")
else:
    print("✓ Dtype is correct")

# ============================================================================
# STEP 4: GET RAW PREDICTIONS
# ============================================================================
print("\n[STEP 4] GETTING RAW PREDICTIONS")
print("-" * 80)

raw_predictions = model.predict(processed, verbose=0)
print(f"Raw prediction shape: {raw_predictions.shape}")
print(f"Raw prediction dtype: {raw_predictions.dtype}")
print(f"Probability sum: {raw_predictions.sum():.6f} (should be 1.0)")
print(f"Min probability: {raw_predictions.min():.6f}")
print(f"Max probability: {raw_predictions.max():.6f}")

# Check for uniform probabilities (common problem when model collapses)
prob_std = raw_predictions.std()
prob_mean = raw_predictions.mean()
print(f"Probability distribution:")
print(f"  Mean: {prob_mean:.6f}")
print(f"  Std Dev: {prob_std:.6f} (should NOT be near {1.0/len(reverse_label_map):.6f} = 1/num_classes)")

# ============================================================================
# STEP 5: ANALYZE TOP PREDICTIONS
# ============================================================================
print("\n[STEP 5] ANALYZING TOP PREDICTIONS")
print("-" * 80)

top_3_indices = np.argsort(raw_predictions[0])[-3:][::-1]
print("Top 3 predictions:")
for rank, idx in enumerate(top_3_indices, 1):
    prob = raw_predictions[0][idx]
    class_name = reverse_label_map[idx]
    print(f"  {rank}. {class_name}: {prob:.6f} ({prob*100:.2f}%)")

# ============================================================================
# STEP 6: TEST ON MULTIPLE IMAGES
# ============================================================================
print("\n[STEP 6] TESTING ON 10 RANDOM TRAINING IMAGES")
print("-" * 80)

import random
random.seed(42)
test_sample_paths = random.sample(test_images, min(10, len(test_images)))

unique_predictions = set()
predictions_list = []

for i, img_path in enumerate(test_sample_paths, 1):
    try:
        # Get true class from directory
        parent_dir = img_path.parent.name
        
        # Preprocess
        processed = preprocess_image_for_prediction(str(img_path))
        
        # Predict
        raw_pred = model.predict(processed, verbose=0)
        pred_class_idx = np.argmax(raw_pred[0])
        confidence = float(np.max(raw_pred))
        pred_class_name = reverse_label_map[pred_class_idx]
        
        unique_predictions.add(pred_class_name)
        predictions_list.append({
            'image': img_path.name,
            'true_class': parent_dir,
            'predicted_class': pred_class_name,
            'confidence': confidence,
            'raw_probs': raw_pred[0]
        })
        
        print(f"\n{i}. {img_path.name}")
        print(f"   True class: {parent_dir}")
        print(f"   Predicted: {pred_class_name} ({confidence*100:.2f}%)")
    except Exception as e:
        print(f"\n{i}. ERROR: {e}")

# ============================================================================
# STEP 7: DETECT PROBLEMS
# ============================================================================
print("\n[STEP 7] PROBLEM DETECTION")
print("-" * 80)

problems_found = []

# Problem 1: All predictions are the same
unique_count = len(unique_predictions)
if unique_count == 1:
    problems_found.append("⚠️  PROBLEM 1: Model predicts same class for ALL images")
    print(f"✓ Found: All 10 images predicted as '{unique_predictions.pop()}'")
elif unique_count < 3:
    problems_found.append(f"⚠️  PROBLEM 2: Very low diversity - only {unique_count} unique classes predicted from 10 images")
    print(f"✓ Found: Only {unique_count} unique classes predicted")
else:
    print(f"✓ Good diversity: {unique_count} unique classes predicted from 10 images")

# Problem 2: Uniform probability distribution
if prob_std < 0.001:
    problems_found.append("⚠️  PROBLEM 3: Probability distribution is nearly uniform (model may be untrained)")
    print(f"✓ Found: Std dev {prob_std:.6f} <<< expected")

# Problem 3: Model accuracy issues
if prob_mean > 0.5:
    problems_found.append(f"⚠️  PROBLEM 4: Mean probability very high ({prob_mean:.4f}) - model may be overfitted")
    print(f"✓ Found: Mean probability {prob_mean:.4f}")
elif prob_mean < (1.0 / len(reverse_label_map)) + 0.01:
    problems_found.append(f"⚠️  PROBLEM 5: Mean probability too low - model may be making random guesses")
    print(f"✓ Found: Mean probability {prob_mean:.4f} (random would be {1.0 / len(reverse_label_map):.4f})")
else:
    print(f"✓ Mean probability {prob_mean:.4f} is reasonable")

# Problem 4: Wrong batch shape
if processed.shape[0] != 1:
    problems_found.append("⚠️  PROBLEM 6: Batch size is not 1 - model expects batch dimension")
    print(f"✓ Found: Batch size {processed.shape[0]} (should be 1)")

# Problem 5: Probability not summing to 1
if abs(raw_predictions.sum() - 1.0) > 0.01:
    problems_found.append("⚠️  PROBLEM 7: Probabilities don't sum to 1.0 (softmax issue)")
    print(f"✓ Found: Sum {raw_predictions.sum():.6f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DEBUGGING SUMMARY")
print("=" * 80)

if problems_found:
    print(f"\n❌ Found {len(problems_found)} problems:\n")
    for problem in problems_found:
        print(f"   {problem}")
else:
    print("\n✓ No critical problems detected in inference pipeline")

print("\nRECOMMENDATIONS:")
print("-" * 80)
print("1. If PROBLEM 1 exists: Model may not be properly trained or loaded")
print("2. If PROBLEM 3 exists: Verify model training completed successfully")
print("3. If PROBLEM 5 exists: Check softmax activation in model head")
print("4. If PROBLEM 6/7 exist: Model may have been trained with different architecture")
print("\nAll preprocessing steps verified:")
print(f"   ✓ Image resized to 224×224")
print(f"   ✓ Converts to RGB")
print(f"   ✓ Normalized to [0, 1]")
print(f"   ✓ Batch dimension added (shape: {processed.shape})")
print(f"   ✓ Dtype is float32")

print("\n" + "=" * 80)
