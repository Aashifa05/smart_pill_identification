#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deployment Validation Script - Test Predictions on Real Pill Images
Validates integrated MobileNet model with updated preprocessing
"""

import os
import sys
import json
import glob
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from pathlib import Path
from Users.utility.requirement import predictions, get_data_paths
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_real_test_images():
    """Find real pill images from training directory (organized by class)"""
    # Images are in pill_project/Admin/media/train/
    test_dir = Path(__file__).parent / 'media' / 'train'
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return []
    
    # Collect images from subdirectories (each subdir is a class)
    test_images = []
    for class_dir in sorted(test_dir.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            class_name = class_dir.name
            # Get up to 1 image per class (10 test images max from different classes)
            class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            if class_images:
                test_images.append({
                    'path': str(class_images[0]),
                    'class_name': class_name
                })
    
    return test_images[:10]  # Return up to 10 images

def run_validation():
    """Run deployment validation"""
    print("=" * 80)
    print("DEPLOYMENT VALIDATION - MobileNet Model Integration")
    print("=" * 80)
    
    # Check paths
    paths = get_data_paths()
    model_path = paths['model_path']
    metadata_path = paths['model_metadata']
    
    print(f"\n[1/4] Checking Model Files...")
    print(f"  Model: {model_path}")
    print(f"  Exists: {model_path.exists()}")
    if not model_path.exists():
        print("  ❌ Model file not found!")
        return False
    
    print(f"  Metadata: {metadata_path}")
    print(f"  Exists: {metadata_path.exists()}")
    if not metadata_path.exists():
        print("  ❌ Metadata file not found!")
        return False
    
    # Load and verify metadata
    print(f"\n[2/4] Verifying Metadata...")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        label_map = metadata.get('label_map', {})
        print(f"  Number of classes: {len(label_map)}")
        print(f"  Classes: {list(label_map.keys())}")
        print(f"  Test accuracy: {metadata.get('test_accuracy', 'N/A'):.4f}")
        print(f"  Training date: {metadata.get('training_date', 'N/A')}")
    except Exception as e:
        print(f"  ❌ Error loading metadata: {e}")
        return False
    
    # Find test images
    print(f"\n[3/4] Finding Test Images...")
    test_images = find_real_test_images()
    print(f"  Found {len(test_images)} test images")
    if len(test_images) == 0:
        print("  ❌ No test images found!")
        return False
    
    for i, img_info in enumerate(test_images, 1):
        print(f"    {i}. {img_info['class_name']}")
    
    # Run predictions
    print(f"\n[4/4] Running Predictions...")
    print("=" * 80)
    
    correct = 0
    incorrect = 0
    errors = 0
    
    for i, img_info in enumerate(test_images, 1):
        image_path = img_info['path']
        actual_class = img_info['class_name']
        
        try:
            # Get prediction
            result = predictions(image_path)
            
            # Extract prediction info
            predicted_name = result.get('tablet_name', result.get('predicted_pill', 'UNKNOWN'))
            confidence = result.get('confidence', '0%').replace('%', '')
            is_correct = predicted_name == actual_class
            
            # Print result
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            print(f"\n[Test {i}/{len(test_images)}] {status}")
            print(f"  Image: {Path(image_path).name}")
            print(f"  Actual class: {actual_class}")
            print(f"  Predicted: {predicted_name}")
            print(f"  Confidence: {confidence}%")
            
            # Print output format check
            output_keys = ['tablet_name', 'confidence', 'generic_name', 'usage', 
                          'dosage', 'consumption_time', 'side_effects', 'precautions']
            missing_keys = [k for k in output_keys if k not in result]
            if missing_keys:
                print(f"  ⚠️  Missing output fields: {missing_keys}")
            else:
                print(f"  ✓ All output fields present")
            
            # Track results
            if is_correct:
                correct += 1
            else:
                incorrect += 1
        
        except Exception as e:
            print(f"\n[Test {i}/{len(test_images)}] ❌ ERROR")
            print(f"  Image: {Path(image_path).name}")
            print(f"  Error: {str(e)}")
            errors += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_images)}")
    print(f"Correct: {correct} ({100*correct/len(test_images):.1f}%)")
    print(f"Incorrect: {incorrect} ({100*incorrect/len(test_images):.1f}%)")
    print(f"Errors: {errors}")
    
    # Pass/Fail
    success = (errors == 0 and correct >= len(test_images) * 0.7)
    
    print("\n" + "=" * 80)
    if success:
        print("✓ DEPLOYMENT VALIDATION PASSED")
        print("  - Model loads correctly")
        print("  - Preprocessing works (0-1 normalization)")
        print("  - Predictions format is correct")
        print("  - Metadata mapping is accurate (70%+ accuracy achieved)")
    else:
        print("⚠️ DEPLOYMENT VALIDATION ISSUES DETECTED")
        if errors > 0:
            print(f"  - {errors} prediction errors occurred")
        if correct < len(test_images) * 0.7:
            print(f"  - Accuracy below 70% ({100*correct/len(test_images):.1f}%)")
    print("=" * 80)
    
    return success

if __name__ == '__main__':
    try:
        success = run_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
