#!/usr/bin/env python
"""Test script to verify model loading"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import time

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from pathlib import Path

print("="*70)
print("PILL DETECTION MODEL LOADING TEST")
print("="*70)

# Test with .keras format
model_path = Path("media/pilldata/model.keras")

if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

print(f"\n[1] Model file: {model_path}")
print(f"    File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

print(f"\n[2] Loading model...")
start_time = time.time()

try:
    from tensorflow.keras.models import load_model
    model = load_model(str(model_path))
    load_time = time.time() - start_time
    print(f"    SUCCESS - Loaded in {load_time:.2f}s")
    print(f"    Input shape: {model.input_shape}")
    print(f"    Output shape: {model.output_shape}")
    print(f"    Total parameters: {model.count_params():,}")
except Exception as e:
    print(f"    FAILED: {e}")
    sys.exit(1)

print(f"\n[3] Testing prediction...")
try:
    import numpy as np
    dummy_input = np.random.rand(1, 224, 224, 3).astype('float32')
    pred = model.predict(dummy_input, verbose=0)
    print(f"    SUCCESS - Prediction shape: {pred.shape}")
    print(f"    Top 3 predictions: {np.argsort(pred[0])[-3:][::-1]}")
except Exception as e:
    print(f"    FAILED: {e}")
    sys.exit(1)

print(f"\n[4] Checking model for use in predictions...")
try:
    from Users.utility.requirement import predictions
    print(f"    Predictions module imported successfully")
except Exception as e:
    print(f"    WARNING: Could not import predictions module: {e}")

print("\n" + "="*70)
print("[OK] MODEL IS READY FOR USE")
print("="*70)
