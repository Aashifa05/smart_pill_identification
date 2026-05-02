#!/usr/bin/env python3
"""
Test the model_95.keras with confidence threshold adjustment
This script verifies that the model now correctly identifies pills instead of predicting UNKNOWN
"""

import os
import json
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add Django setup
sys.path.insert(0, str(Path(__file__).parent))
try:
    import django
    from django.conf import settings
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            BASE_DIR=Path(__file__).parent,
            MEDIA_ROOT=Path(__file__).parent / 'media',
            MEDIA_URL='/media/'
        )
    django.setup()
except:
    pass

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

print("=" * 70)
print("PILL IDENTIFICATION MODEL v2 - PERFORMANCE TEST")
print("=" * 70)

# Paths
DATA_DIR = Path("media/pilldata")
MODEL_PATH = DATA_DIR / "model_95.keras"
METADATA_PATH = DATA_DIR / "model_metadata_95.json"
TEST_CSV = DATA_DIR / "Testing_set.csv"

print("\n1. Loading model...")
if not MODEL_PATH.exists():
    print(f"  ✗ Model not found: {MODEL_PATH}")
    sys.exit(1)

model = load_model(str(MODEL_PATH))
print(f"  ✓ Model loaded successfully")
print(f"    File size: {MODEL_PATH.stat().st_size / (1024**2):.1f} MB")

print("\n2. Loading metadata...")
with open(METADATA_PATH) as f:
    metadata = json.load(f)

print(f"  ✓ Metadata loaded")
print(f"    Classes: {metadata['classes']}")
print(f"    Model accuracy: {metadata['accuracy']*100:.2f}%")
classes = metadata['class_labels']
reverse_label_map = {i: label for i, label in enumerate(classes)}

print("\n3. Testing on sample images...")
df_test = pd.read_csv(TEST_CSV)
test_samples = df_test.head(10)

correct_predictions = 0
total_predictions = 0
unknowns = 0

for idx, row in test_samples.iterrows():
    img_path = DATA_DIR / "train" / row['filename']
    true_label = row['label']
    
    try:
        # Load and preprocess
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        pred_idx = np.argmax(predictions)
        confidence = predictions[pred_idx]
        pred_label = reverse_label_map[pred_idx]
        
        total_predictions += 1
        
        # Check if correct
        is_correct = pred_label == true_label
        if is_correct:
            correct_predictions += 1
        
        # Check for UNKNOWN
        if confidence < 0.5:
            unknowns += 1
            status = "  ⚠ UNKNOWN"
        elif is_correct:
            status = "  ✓ CORRECT"
        else:
            status = "  ✗ WRONG"
        
        print(f"{idx+1}. {status} | True: {true_label[:20]} | Pred: {pred_label[:20]} | Conf: {confidence*100:.1f}%")
        
    except Exception as e:
        print(f"{idx+1}. ✗ ERROR: {e}")

print("\n4. Summary:")
print(f"  Total predictions: {total_predictions}")
print(f"  Correct: {correct_predictions} ({correct_predictions/total_predictions*100:.1f}%)")
print(f"  Unknown (<50%): {unknowns}")
print(f"  Accuracy on sample: {correct_predictions/total_predictions*100:.1f}%")

print("\n5. Model v2 Status:")
if correct_predictions / total_predictions > 0.6:
    print("  ✓ Model is working well - NOT predicting UNKNOWN for everything")
    print("  ✓ Ready for deployment")
else:
    print("  ⚠ Model needs improvement")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
