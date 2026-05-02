#!/usr/bin/env python
"""
COMPREHENSIVE SYSTEM VERIFICATION TEST
Tests all critical components of the Pill Detection System
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\n" + "="*80)
print("PILL DETECTION & ANALYSIS SYSTEM - COMPREHENSIVE VERIFICATION".center(80))
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

test_results = []

# Test 1: Django Setup
print("[1/6] Testing Django Setup...")
try:
    import django
    django.setup()
    test_results.append(("Django Setup", "PASS"))
    print("      ✓ Django setup complete")
except Exception as e:
    test_results.append(("Django Setup", f"FAIL: {e}"))
    print(f"      ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Database Models
print("\n[2/6] Testing Database Models...")
try:
    from Users.models import UserRegisteredTable, PillInfo
    from Admin.models import AdminModel
    
    # Check pill information
    pill_count = PillInfo.objects.count()
    print(f"      ✓ Database models loaded")
    print(f"        - Pills in database: {pill_count}/10")
    
    if pill_count >= 10:
        test_results.append(("Database Models", "PASS"))
    else:
        test_results.append(("Database Models", f"WARNING: Only {pill_count}/10 pills"))
except Exception as e:
    test_results.append(("Database Models", f"FAIL: {e}"))
    print(f"      ✗ Failed: {e}")

# Test 3: Model Files
print("\n[3/6] Testing Model Files...")
try:
    from Users.utility.requirement import get_data_paths
    paths = get_data_paths()
    
    model_path = paths['model_path']
    metadata_path = paths['model_metadata']
    
    model_exists = model_path.exists()
    metadata_exists = metadata_path.exists()
    
    print(f"      ✓ Paths configured")
    print(f"        - Model file exists: {model_exists} ({model_path.name})")
    print(f"        - Metadata file exists: {metadata_exists}")
    
    if model_exists and metadata_exists:
        test_results.append(("Model Files", "PASS"))
    else:
        test_results.append(("Model Files", "FAIL: Missing model or metadata"))
except Exception as e:
    test_results.append(("Model Files", f"FAIL: {e}"))
    print(f"      ✗ Failed: {e}")

# Test 4: Model Loading
print("\n[4/6] Testing Model Loading...")
try:
    from tensorflow.keras.models import load_model
    import time
    
    start = time.time()
    model = load_model(str(model_path))
    load_time = time.time() - start
    
    print(f"      ✓ Model loaded successfully in {load_time:.2f}s")
    print(f"        - Input shape: {model.input_shape}")
    print(f"        - Output shape: {model.output_shape}")
    print(f"        - Parameters: {model.count_params():,}")
    
    test_results.append(("Model Loading", "PASS"))
except Exception as e:
    test_results.append(("Model Loading", f"FAIL: {e}"))
    print(f"      ✗ Failed: {e}")

# Test 5: Prediction Function
print("\n[5/6] Testing Prediction Function...")
try:
    from Users.utility.requirement import predictions
    from PIL import Image
    import numpy as np
    
    # Create a test image
    test_img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    test_img_path = Path('_temp_test_img.jpg')
    test_img.save(test_img_path)
    
    # Run prediction
    result = predictions(str(test_img_path))
    
    # Verify result structure
    required_keys = ['predicted_pill', 'confidence', 'confidence_percentage']
    missing_keys = [k for k in required_keys if k not in result]
    
    if not missing_keys:
        print(f"      ✓ Prediction function works")
        print(f"        - Predicted: {result['predicted_pill']}")
        print(f"        - Confidence: {result['confidence_percentage']}")
        test_results.append(("Prediction Function", "PASS"))
    else:
        test_results.append(("Prediction Function", f"FAIL: Missing {missing_keys}"))
    
    # Clean up
    test_img_path.unlink()
    
except Exception as e:
    test_results.append(("Prediction Function", f"FAIL: {e}"))
    print(f"      ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Views Import
print("\n[6/6] Testing Django Views...")
try:
    from Users.views import userRegister, userLoginCheck, userPrediction
    from Admin.views import adminLogin, adminLogout, adminClassificationView
    
    print(f"      ✓ All views imported successfully")
    print(f"        - Users: userRegister, userLoginCheck, userPrediction")
    print(f"        - Admin: adminLogin, adminLogout, adminClassificationView")
    
    test_results.append(("Django Views", "PASS"))
except Exception as e:
    test_results.append(("Django Views", f"FAIL: {e}"))
    print(f"      ✗ Failed: {e}")

# Print Summary
print("\n" + "="*80)
print("TEST SUMMARY".center(80))
print("="*80)

pass_count = sum(1 for _, result in test_results if result == "PASS")
total_count = len(test_results)

for test_name, result in test_results:
    status = "✓ PASS" if result == "PASS" else "✗ " + result
    print(f"  {test_name:.<45} {status}")

print("="*80)
print(f"Results: {pass_count}/{total_count} tests passed".center(80))

if pass_count == total_count:
    print("\n✓ SYSTEM IS FULLY OPERATIONAL".center(80))
    print("\nYou can now:".center(80))
    print("  1. Register users and manage accounts".center(80))
    print("  2. Upload pill images for classification".center(80))
    print("  3. Get pill information (usage, dosage, side effects)".center(80))
    print("  4. Administer the system".center(80))
else:
    print(f"\n⚠ {total_count - pass_count} TESTS NEED ATTENTION".center(80))

print("="*80 + "\n")
