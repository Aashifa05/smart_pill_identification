#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test of the multi-feature pill classifier
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.multi_feature_pill_classifier import (
    MultiFeaturePillClassifier,
    format_comprehensive_report
)
from django.conf import settings
import json

print("="*70)
print("MULTI-FEATURE PILL CLASSIFIER - QUICK TEST")
print("="*70)

# Check available model files
model_dir = Path(settings.BASE_DIR) / 'media' / 'pilldata'
print(f"\nChecking models in: {model_dir}\n")

model_files = list(model_dir.glob('*.keras')) + list(model_dir.glob('*.h5'))
for f in model_files:
    size_mb = f.stat().st_size / (1024*1024)
    print(f"✓ {f.name:40s} ({size_mb:6.1f} MB)")

# Try to load the best available model
print("\n" + "="*70)
print("Attempting to load classifier...")
print("="*70)

# Try imprint_robust first, then fallback to others
model_candidates = [
    ('model_imprint_robust.h5', 'model_imprint_robust_metadata.json'),
    ('model.keras', 'model_metadata.json'),
    ('model_95.keras', 'model_metadata_95.json'),
]

classifier = None
loaded_model = None

for model_file, metadata_file in model_candidates:
    model_path = model_dir / model_file
    metadata_path = model_dir / metadata_file
    
    if model_path.exists() and metadata_path.exists():
        try:
            print(f"\nLoading: {model_file}")
            classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))
            loaded_model = model_file
            print(f"✓ Successfully loaded {model_file}")
            break
        except Exception as e:
            print(f"✗ Failed to load {model_file}: {str(e)[:100]}")
            continue

if classifier is None:
    print("\n✗ ERROR: Could not load any model!")
    print("\nTo train the imprint-robust model, run:")
    print("  python train_imprint_robust.py")
    sys.exit(1)

print("\n" + "="*70)
print(f"✓ CLASSIFIER LOADED: {loaded_model}")
print("="*70)

# Check for test images
test_image_dirs = [
    model_dir / 'test_images',
    Path(settings.BASE_DIR) / 'media' / 'test_images',
    Path(settings.BASE_DIR) / 'media' / 'test',
]

test_images = []
for test_dir in test_image_dirs:
    if test_dir.exists():
        test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        if test_images:
            print(f"\n✓ Found {len(test_images)} test images in {test_dir.name}")
            break

if not test_images:
    print("\n⚠ No test images found.")
    print("\nTo test with an image, you can either:")
    print("  1. Add images to media/test_images/")
    print("  2. Provide an image path")
    print("\nExample usage:")
    print("  result = classifier.predict('path/to/pill/image.jpg')")
    print("  print(result)")
else:
    # Test with first image
    test_image = test_images[0]
    print(f"\n{'='*70}")
    print(f"Testing with: {test_image.name}")
    print(f"{'='*70}")
    
    try:
        result = classifier.predict(str(test_image))
        
        print(f"\n✓ Classification successful!")
        print(f"\nResults:")
        print(f"  Status:        {result['status']}")
        print(f"  Pill Name:     {result['pill_name']}")
        print(f"  Confidence:    {result['confidence']:.1%}")
        print(f"  Threshold:     {result['adjusted_threshold']:.1%}")
        print(f"  Has Imprints:  {result['analysis']['has_visible_imprints']}")
        
        print(f"\nFeatures Extracted:")
        print(f"  ✓ Shape features: {list(result['features']['shape'].keys())}")
        print(f"  ✓ Color features: {list(result['features']['color'].keys())}")
        print(f"  ✓ Size features:  {list(result['features']['size'].keys())}")
        print(f"  ✓ Texture features: {list(result['features']['texture'].keys())}")
        print(f"  ✓ Imprint features: {list(result['features']['imprint'].keys())}")
        
        print(f"\nClassification Basis:")
        print(f"  {result['analysis']['classification_basis']['description']}")
        
        print(f"\nTop 5 Predictions:")
        for pred in result['top_5_predictions'][:5]:
            print(f"  {pred['rank']}. {pred['pill_name']:35s} {pred['confidence']:.1%}")
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("✓ TEST COMPLETE")
print("="*70)

print(f"\nQuick Reference:")
print(f"  • Classifier class: Users.utility.multi_feature_pill_classifier.MultiFeaturePillClassifier")
print(f"  • Guide: MULTI_FEATURE_CLASSIFICATION_GUIDE.md")
print(f"  • Quick ref: QUICK_REFERENCE_MULTI_FEATURE.md")
print(f"  • Examples: multi_feature_classification_examples.py")
print(f"  • Tests: test_multi_feature_classifier.py")
print()
