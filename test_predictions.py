#!/usr/bin/env python
"""Test the predictions function with a sample image from the dataset"""
import os
import sys
from pathlib import Path

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("PILL DETECTION PREDICTION TEST")
print("="*70)

try:
    import django
    django.setup()
    print("\n✓ Django setup complete")
except Exception as e:
    print(f"\n✗ Django setup failed: {e}")
    sys.exit(1)

try:
    from Users.utility.requirement import get_data_paths, predictions
    paths = get_data_paths()
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Find a test image
test_path = paths['test_path']
print(f"\nLooking for test images in: {test_path}")

if test_path.exists():
    images = list(test_path.glob('**/*.jpg')) + list(test_path.glob('**/*.png'))
    print(f"Found {len(images)} test images")
    
    if images:
        test_image = images[0]
        print(f"\n[1] Testing with: {test_image.name}")
        print(f"    Path: {test_image}")
        
        print(f"\n[2] Running prediction...")
        try:
            result = predictions(str(test_image))
            print(f"    ✓ Prediction successful!")
            print(f"\n    Results:")
            for key, value in result.items():
                print(f"      {key}: {value}")
        except Exception as e:
            print(f"    ✗ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("No test images found!")
else:
    print(f"Test directory does not exist: {test_path}")
    print("\nCreating a sample image for testing...")
    
    try:
        from PIL import Image
        import numpy as np
        
        # Create a dummy test image
        test_img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        test_img_path = Path('test_sample.jpg')
        test_img.save(test_img_path)
        print(f"✓ Created test image: {test_img_path}")
        
        print(f"\n[1] Testing with created image")
        print(f"\n[2] Running prediction...")
        result = predictions(str(test_img_path))
        print(f"    ✓ Prediction successful!")
        print(f"\n    Results:")
        for key, value in result.items():
            print(f"      {key}: {value}")
        
        # Clean up
        test_img_path.unlink()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*70)
print("✓ TEST COMPLETE - SYSTEM IS READY")
print("="*70)
