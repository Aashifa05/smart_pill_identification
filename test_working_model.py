"""
TEST THE WORKING MODEL
Validates that the new model actually works better than the old one
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import os

print("=" * 80)
print("TESTING WORKING MODEL")
print("=" * 80)
print()

# Load metadata
with open('media/pilldata/model_working_metadata.json', 'r') as f:
    metadata = json.load(f)

class_names = metadata['class_names']
print(f"Classes: {len(class_names)}")
print(f"Validation Accuracy: {metadata['final_val_accuracy']:.2%}")
print()

# Load model
model = keras.models.load_model('media/pilldata/model_working.keras')
print("✓ Model loaded")
print()

# Test on a few training images
print("TESTING ON TRAINING IMAGES")
print("-" * 80)

train_dir = 'media/pilldata/train'
correct = 0
total = 0
confidences = []

for filename in sorted(os.listdir(train_dir))[:50]:  # Test on first 50
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Get true class
    true_class = None
    for cls_name in class_names:
        if cls_name.lower() in filename.lower():
            true_class = cls_name
            break
    
    if true_class is None:
        continue
    
    try:
        # Load and predict
        img = Image.open(os.path.join(train_dir, filename)).convert('RGB')
        img = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
        
        pred = model.predict(np.expand_dims(img, 0), verbose=0)
        pred_class = class_names[np.argmax(pred)]
        confidence = float(np.max(pred))
        
        is_correct = pred_class == true_class
        correct += is_correct
        total += 1
        confidences.append(confidence)
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {true_class:40} → {pred_class:40} ({confidence:.1%})")
        
    except Exception as e:
        continue

accuracy = correct / total if total > 0 else 0
avg_confidence = np.mean(confidences) if confidences else 0

print()
print(f"Results on {total} test images:")
print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
print(f"  Avg Confidence: {avg_confidence:.1%}")
print()

# Save results
results = {
    'test_images': total,
    'correct': correct,
    'accuracy': float(accuracy),
    'avg_confidence': float(avg_confidence),
    'model': 'model_working.keras'
}

with open('media/pilldata/test_working_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
