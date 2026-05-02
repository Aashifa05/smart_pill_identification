"""Test and compare existing models"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import os

print("\n" + "=" * 80)
print("TESTING & COMPARING PILL MODELS")
print("=" * 80 + "\n")

# Load old model
print("Loading models...")
try:
    old_model = keras.models.load_model('media/pilldata/model_anti_overfit.keras')
    print("✓ Old model loaded: model_anti_overfit.keras")
except Exception as e:
    print(f"✗ Error loading old model: {e}")
    old_model = None

try:
    new_model = keras.models.load_model('media/pilldata/model_enhanced.keras')
    print("✓ New model loaded: model_enhanced.keras")
except Exception as e:
    print(f"✗ Error loading new model: {e}")
    new_model = None

CLASS_NAMES = [
    'Amoxicillin 500 MG', 'Atomoxetine 25 MG', 'Calcitriol 0.00025 MG',
    'Oseltamivir 45 MG', 'Ramipril 5 MG', 'apixaban 2.5 MG',
    'aprepitant 80 MG', 'benzonatate 100 MG', 'carvedilol 3.125 MG',
    'celecoxib 200 MG', 'duloxetine 30 MG', 'eltrombopag 25 MG',
    'montelukast 10 MG', 'mycophenolate mofetil 250 MG',
    'pantoprazole 40 MG', 'pitavastatin 1 MG', 'prasugrel 10 MG',
    'saxagliptin 5 MG', 'sitagliptin 50 MG', 'tadalafil 5 MG'
]

train_dir = 'media/pilldata/train'
test_images = []

print("\nPreparing test images...")
for filename in sorted(os.listdir(train_dir))[:100]:  # Test on 100 images
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    true_class = None
    for cls_name in CLASS_NAMES:
        if cls_name.lower() in filename.lower():
            true_class = cls_name
            break
    
    if true_class is None:
        continue
    
    try:
        img_path = os.path.join(train_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
        test_images.append((img, true_class, filename))
    except:
        continue

print(f"Found {len(test_images)} test images\n")

# Test models
if old_model:
    print("=" * 80)
    print("OLD MODEL: model_anti_overfit.keras")
    print("=" * 80)
    
    old_correct = 0
    old_confidences = []
    
    for img, true_class, filename in test_images:
        pred = old_model.predict(np.expand_dims(img, 0), verbose=0)
        pred_class = CLASS_NAMES[np.argmax(pred)]
        confidence = float(np.max(pred))
        
        if pred_class == true_class:
            old_correct += 1
        old_confidences.append(confidence)
    
    old_acc = old_correct / len(test_images) if test_images else 0
    old_avg_conf = np.mean(old_confidences) if old_confidences else 0
    
    print(f"Accuracy: {old_acc:.1%} ({old_correct}/{len(test_images)})")
    print(f"Avg Confidence: {old_avg_conf:.1%}\n")

if new_model:
    print("=" * 80)
    print("NEW MODEL: model_enhanced.keras")
    print("=" * 80)
    
    new_correct = 0
    new_confidences = []
    
    for img, true_class, filename in test_images:
        pred = new_model.predict(np.expand_dims(img, 0), verbose=0)
        pred_class = CLASS_NAMES[np.argmax(pred)]
        confidence = float(np.max(pred))
        
        if pred_class == true_class:
            new_correct += 1
        new_confidences.append(confidence)
    
    new_acc = new_correct / len(test_images) if test_images else 0
    new_avg_conf = np.mean(new_confidences) if new_confidences else 0
    
    print(f"Accuracy: {new_acc:.1%} ({new_correct}/{len(test_images)})")
    print(f"Avg Confidence: {new_avg_conf:.1%}\n")

if old_model and new_model:
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    improvement = (new_acc - old_acc) / old_acc * 100 if old_acc > 0 else 0
    print(f"Accuracy change: {old_acc:.1%} → {new_acc:.1%} ({improvement:+.1f}%)")
    print(f"Confidence change: {old_avg_conf:.1%} → {new_avg_conf:.1%} ({(new_avg_conf - old_avg_conf):.1%})\n")
    
    if new_acc > old_acc:
        print(f"✓ New model is BETTER (+{improvement:.1f}%)")
    elif new_acc < old_acc:
        print(f"✗ New model is WORSE ({improvement:.1f}%)")
    else:
        print("= Models are EQUAL")

print("\n" + "=" * 80)
