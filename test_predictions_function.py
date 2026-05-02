#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the Actual predictions() Function
========================================

This tests the EXACT SAME FUNCTION that the deployed app uses.

"""

import os
import sys
import numpy as np
from pathlib import Path
import random

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.requirement import predictions, get_data_paths

print("=" * 80)
print("TESTING ACTUAL predictions() FUNCTION")
print("=" * 80)

paths = get_data_paths()
train_path = paths['train_path']

# Find test images
test_images = list(train_path.glob("**/*.jpg")) + list(train_path.glob("**/*.png"))
if not test_images:
    print("❌ ERROR: No test images found")
    sys.exit(1)

print(f"Found {len(test_images)} test images")

# Test 10 random images
test_sample_paths = random.sample(test_images, min(10, len(test_images)))

print(f"\nTesting {len(test_sample_paths)} images with predictions() function:")
print("-" * 80)

predictions_dict = {}
unique_classes = set()

for i, img_path in enumerate(test_sample_paths, 1):
    try:
        print(f"\n{i}. {img_path.name}")
        
        # Call the actual predictions function
        result = predictions(str(img_path), confidence_threshold=0.50)
        
        predicted = result.get('predicted_label', 'ERROR')
        confidence = result.get('confidence', 0)
        is_confident = result.get('is_confident', False)
        
        print(f"   Predicted: {predicted}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   Is confident (>50%): {is_confident}")
        print(f"   Top 3 candidates: {result.get('top_3_candidates', [])}")
        
        # Track predictions
        unique_classes.add(predicted)
        
        if predicted not in predictions_dict:
            predictions_dict[predicted] = 0
        predictions_dict[predicted] += 1
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

# Analyze results
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"\nTotal unique classes predicted: {len(unique_classes)}")
print(f"Prediction distribution:")
for class_name, count in sorted(predictions_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {count}/10")

if len(unique_classes) == 1:
    print("\n❌ CRITICAL ISSUE: Model is predicting the SAME CLASS for ALL images!")
    print(f"   All images predicted as: {list(unique_classes)[0]}")
elif len(unique_classes) < 3:
    print(f"\n⚠️  WARNING: Very low diversity - only {len(unique_classes)} unique classes from 10 images")
else:
    print(f"\n✓ Good diversity: {len(unique_classes)} unique classes predicted")

print("\n" + "=" * 80)
