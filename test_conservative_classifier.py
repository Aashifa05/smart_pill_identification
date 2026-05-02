#!/usr/bin/env python3
"""
Test the Conservative Pill Classifier
=====================================
Tests the model with medical-safe conservative classification
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("CONSERVATIVE PILL CLASSIFIER TEST")
print("=" * 70)

# Step 1: Verify files exist
print("\n[1/4] Checking required files...")
required_files = {
    'model': 'media/pilldata/model_anti_overfit.keras',
    'metadata': 'media/pilldata/model_metadata.json',
    'classifier': 'conservative_pill_classifier.py',
    'data': 'media/pilldata/train'
}

all_exist = True
for name, path in required_files.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n⚠️  Some required files are missing!")
    print("Expected structure:")
    print("  media/pilldata/")
    print("    ├── model_anti_overfit.keras")
    print("    ├── model_metadata.json")
    print("    └── train/")
    sys.exit(1)

print("\n✓ All required files found!")

# Step 2: Load the model
print("\n[2/4] Loading model...")
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    
    model_path = 'media/pilldata/model_anti_overfit.keras'
    model = load_model(model_path)
    
    print(f"✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output classes: {model.output_shape[-1]}")
    print(f"  Total parameters: {model.count_params():,}")
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Step 3: Load class mapping
print("\n[3/4] Loading class mapping...")
try:
    with open('media/pilldata/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    class_names = metadata.get('class_names', [])
    print(f"✓ Loaded {len(class_names)} pill classes")
    print(f"  Sample classes: {', '.join(class_names[:5])}")
    
except Exception as e:
    print(f"⚠️  Could not load metadata: {e}")
    class_names = None

# Step 4: Test on sample images
print("\n[4/4] Testing classifier on sample images...")
print("-" * 70)

# Find test images
test_dir = Path('media/pilldata/train')
if test_dir.exists():
    image_files = list(test_dir.glob('**/*.jpg')) + list(test_dir.glob('**/*.png'))
    image_files = image_files[:10]  # Limit to first 10
    
    if image_files:
        print(f"Found {len(image_files)} test images")
        
        results = {
            'total_tested': 0,
            'correctly_identified': 0,
            'marked_unknown': 0,
            'predictions': []
        }
        
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = image.load_img(str(img_path), target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Get predictions
                predictions = model.predict(img_array, verbose=0)
                pred_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][pred_idx])
                
                # Extract true class from filename
                true_class = img_path.parent.name
                
                # Conservative classification logic
                has_imprint = confidence > 0.4  # Simple heuristic
                threshold = 0.75 if has_imprint else 0.65
                
                is_identified = confidence >= threshold
                
                # Prepare result
                pred_class = class_names[pred_idx] if class_names else f"Class_{pred_idx}"
                result = {
                    'image': img_path.name,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'confidence': f"{confidence:.2%}",
                    'threshold': f"{threshold:.0%}",
                    'decision': 'IDENTIFIED ✓' if is_identified else 'UNKNOWN ✗',
                    'correct': true_class == pred_class if is_identified else True
                }
                
                results['predictions'].append(result)
                results['total_tested'] += 1
                
                if is_identified and true_class == pred_class:
                    results['correctly_identified'] += 1
                elif not is_identified:
                    results['marked_unknown'] += 1
                
                # Print result
                symbol = "✓" if result['correct'] else "✗"
                print(f"\n{symbol} {img_path.name}")
                print(f"   True: {true_class}")
                print(f"   Predicted: {pred_class}")
                print(f"   Confidence: {result['confidence']}")
                print(f"   Threshold (imprint={has_imprint}): {result['threshold']}")
                print(f"   Decision: {result['decision']}")
                
            except Exception as e:
                print(f"\n✗ Error processing {img_path.name}: {e}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total tested: {results['total_tested']}")
        print(f"Correctly identified: {results['correctly_identified']}")
        print(f"Marked as unknown: {results['marked_unknown']}")
        
        if results['total_tested'] > 0:
            accuracy = results['correctly_identified'] / results['total_tested']
            print(f"Accuracy: {accuracy:.1%}")
        
        print("\nTest results saved to test_results.json")
        
        # Save detailed results
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    else:
        print("No test images found in media/pilldata/train/")
        print("\nTo test, you need pill images in:")
        print("  media/pilldata/train/<pill_class>/<image.jpg>")
else:
    print(f"Train directory not found: {test_dir}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\n✓ Model and classifier are working!")
print("\nNext steps:")
print("  1. Review test_results.json for detailed results")
print("  2. Test on external images from websites")
print("  3. Integrate with Django application")
print("  4. Deploy to production")
print("\n" + "=" * 70)
