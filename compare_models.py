#!/usr/bin/env python3
"""
Compare Models: Old vs Enhanced
================================
Test both models to see improvements in:
1. Accuracy on known pills
2. Rejection of unknown pills
3. Confidence calibration
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("MODEL COMPARISON: OLD vs ENHANCED")
print("=" * 80)
print("\nThis test compares:")
print("  • OLD model (model_anti_overfit.keras)")
print("  • NEW model (model_enhanced.keras - when ready)")
print("\nMetrics:")
print("  • Accuracy on known pills")
print("  • Confidence on known pills")
print("  • Rejection rate on unknown pills")
print("  • False positive rate (unknown marked as known)")
print("  • False negative rate (known marked unknown)")

# ============================================================================
# STEP 1: Load Models
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING MODELS")
print("=" * 80)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

models = {}
model_info = {}

# Load old model
old_model_path = 'media/pilldata/model_anti_overfit.keras'
if Path(old_model_path).exists():
    try:
        models['old'] = load_model(old_model_path)
        print(f"✓ Loaded: {old_model_path}")
    except Exception as e:
        print(f"✗ Could not load old model: {e}")
else:
    print(f"✗ Old model not found: {old_model_path}")

# Load new model
new_model_path = 'media/pilldata/model_enhanced.keras'
if Path(new_model_path).exists():
    try:
        models['new'] = load_model(new_model_path)
        print(f"✓ Loaded: {new_model_path}")
    except Exception as e:
        print(f"✗ Could not load new model: {e}")
else:
    print(f"⏳ New model not ready yet: {new_model_path}")
    print("  (Training in progress...)")

# Load metadata
with open('media/pilldata/model_metadata.json', 'r') as f:
    metadata = json.load(f)

class_names = list(metadata.get('label_map', {}).keys())
class_to_idx = {name: i for i, name in enumerate(class_names)}

print(f"\nClass information:")
print(f"  Known classes: {len(class_names)}")
print(f"  Classes: {', '.join(class_names[:3])}...")

# ============================================================================
# STEP 2: Load Test Data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: LOADING TEST DATA")
print("=" * 80)

train_dir = Path('media/pilldata/train')

# Get known pill images
known_images = {name: [] for name in class_names}
unknown_images = []

print("Scanning training data...")

for img_path in sorted(train_dir.glob('*.jpg')) + sorted(train_dir.glob('*.png')):
    filename = img_path.stem
    clean_name = filename.replace(' - Copy', '').strip()
    parts = clean_name.rsplit(' ', 1)
    if parts[-1].isdigit():
        clean_name = ' '.join(parts[:-1])
    
    # Check if it's a known class
    found = False
    for name in class_names:
        if clean_name.lower() == name.lower():
            known_images[name].append(img_path)
            found = True
            break
    
    if not found:
        unknown_images.append(img_path)

# Sample images for testing
known_test_images = []
known_test_labels = []

print(f"\nSampling test images:")
for class_name in class_names:
    images = known_images[class_name][:5]  # Take first 5 of each class
    for img_path in images:
        known_test_images.append((img_path, class_name))
        known_test_labels.append(class_to_idx[class_name])
    print(f"  {class_name:40} {len(images)} images")

unknown_test_images = unknown_images[:30]  # Take first 30 unknown

print(f"\nTest data summary:")
print(f"  Known pill images: {len(known_test_images)}")
print(f"  Unknown pill images: {len(unknown_test_images)}")

# ============================================================================
# STEP 3: Test Known Pills
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: TESTING ON KNOWN PILLS")
print("=" * 80)

results = {
    'known_pills': {},
    'unknown_pills': {},
    'summary': {}
}

for model_name, model in models.items():
    print(f"\nTesting '{model_name}' model on known pills...")
    
    correct = 0
    total = 0
    confidences = []
    false_unknowns = 0
    
    for img_path, true_class in known_test_images:
        try:
            img = image.load_img(str(img_path), target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])
            pred_class = class_names[pred_idx]
            
            total += 1
            confidences.append(confidence)
            
            if pred_class == true_class:
                correct += 1
            else:
                # Check if this is a false negative (marked as different known class)
                pass
            
            # Check if confidence is below unknown threshold (65%)
            if confidence < 0.65:
                false_unknowns += 1
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
    
    accuracy = correct / total if total > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    false_unknown_rate = false_unknowns / total if total > 0 else 0
    
    results['known_pills'][model_name] = {
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'false_unknowns': false_unknowns,
        'false_unknown_rate': false_unknown_rate
    }
    
    print(f"  ✓ Results for '{model_name}' on known pills:")
    print(f"    Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"    Avg Confidence: {avg_confidence:.1%}")
    print(f"    False Unknowns: {false_unknowns} ({false_unknown_rate:.1%})")

# ============================================================================
# STEP 4: Test Unknown Pills
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TESTING ON UNKNOWN PILLS")
print("=" * 80)

for model_name, model in models.items():
    print(f"\nTesting '{model_name}' model on unknown pills...")
    
    unknown_count = 0
    total = 0
    confidences = []
    
    for img_path in unknown_test_images:
        try:
            img = image.load_img(str(img_path), target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])
            
            total += 1
            confidences.append(confidence)
            
            # Unknown pill should have low confidence (<65%)
            if confidence < 0.65:
                unknown_count += 1
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
    
    unknown_rate = unknown_count / total if total > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    false_positive_rate = (total - unknown_count) / total if total > 0 else 0
    
    results['unknown_pills'][model_name] = {
        'correctly_unknown': unknown_count,
        'total': total,
        'unknown_rate': unknown_rate,
        'false_positive_rate': false_positive_rate,
        'avg_confidence': avg_confidence
    }
    
    print(f"  ✓ Results for '{model_name}' on unknown pills:")
    print(f"    Correctly marked unknown: {unknown_count}/{total} ({unknown_rate:.1%})")
    print(f"    False positives (marked as known): {total - unknown_count}")
    print(f"    False positive rate: {false_positive_rate:.1%}")
    print(f"    Avg Confidence: {avg_confidence:.1%}")

# ============================================================================
# STEP 5: Comparison Summary
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

if 'old' in results['known_pills'] and 'new' in results['known_pills']:
    old_acc = results['known_pills']['old']['accuracy']
    new_acc = results['known_pills']['new']['accuracy']
    improvement = (new_acc - old_acc) / old_acc * 100 if old_acc > 0 else 0
    
    print(f"\nKNOWN PILLS (Accuracy):")
    print(f"  Old: {old_acc:.1%}")
    print(f"  New: {new_acc:.1%}")
    print(f"  Improvement: {improvement:+.1f}%")

if 'old' in results['known_pills'] and 'new' in results['known_pills']:
    old_false_unknown = results['known_pills']['old']['false_unknown_rate']
    new_false_unknown = results['known_pills']['new']['false_unknown_rate']
    improvement = (old_false_unknown - new_false_unknown) / old_false_unknown * 100 if old_false_unknown > 0 else 0
    
    print(f"\nKNOWN PILLS (False Unknown Rate - LOWER IS BETTER):")
    print(f"  Old: {old_false_unknown:.1%}")
    print(f"  New: {new_false_unknown:.1%}")
    print(f"  Improvement: {improvement:+.1f}% (reduction)")

if 'old' in results['unknown_pills'] and 'new' in results['unknown_pills']:
    old_fp = results['unknown_pills']['old']['false_positive_rate']
    new_fp = results['unknown_pills']['new']['false_positive_rate']
    improvement = (old_fp - new_fp) / old_fp * 100 if old_fp > 0 else 0
    
    print(f"\nUNKNOWN PILLS (False Positive Rate - LOWER IS BETTER):")
    print(f"  Old: {old_fp:.1%}")
    print(f"  New: {new_fp:.1%}")
    print(f"  Improvement: {improvement:+.1f}% (reduction)")

# Save detailed results
results_path = Path('model_comparison_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✓ Detailed results saved: {results_path}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Review detailed results in model_comparison_results.json")
print("  2. If new model is better, deploy model_enhanced.keras")
print("  3. Test on real external pill images")
print("  4. Adjust confidence thresholds if needed")
print("\n" + "=" * 80)
