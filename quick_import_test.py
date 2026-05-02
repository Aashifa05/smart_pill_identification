#!/usr/bin/env python
"""Quick import test for the requirement module"""
import os
import sys

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')

# Suppress TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Testing imports...")

try:
    import django
    django.setup()
    print("✓ Django setup complete")
except Exception as e:
    print(f"✗ Django setup failed: {e}")
    sys.exit(1)

try:
    from Users.utility.requirement import get_data_paths
    paths = get_data_paths()
    print(f"✓ get_data_paths() works")
    print(f"  Model path: {paths['model_path']}")
    print(f"  Model exists: {paths['model_path'].exists()}")
except Exception as e:
    print(f"✗ get_data_paths() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from Users.utility.requirement import preprocess_image_for_prediction
    print("✓ preprocess_image_for_prediction imported")
except Exception as e:
    print(f"✗ preprocess_image_for_prediction failed: {e}")
    sys.exit(1)

try:
    from Users.utility.requirement import predictions
    print("✓ predictions function imported")
except Exception as e:
    print(f"✗ predictions function failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All imports successful!")
