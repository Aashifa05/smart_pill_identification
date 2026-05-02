#!/usr/bin/env python
"""Simple model diagnostic - evaluate on training data"""

import os
import sys
import re
import logging
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def extract_pill_name(filename):
    """Extract pill name from filename"""
    match = re.match(r'([^0-9]*?)\s+\d+', filename)
    if match:
        return match.group(1).strip()
    
    match = re.match(r'([a-zA-Z0-9]+-?[a-zA-Z0-9]*?)_', filename)
    if match:
        return match.group(1).strip()
    
    match = re.match(r'([a-zA-Z0-9]+-[a-zA-Z0-9]+)', filename)
    if match:
        return match.group(1).strip()
    
    return None

def load_test_data(num_per_class=20):
    """Load test samples for evaluation"""
    logger.info(f"Loading test data ({num_per_class} samples per class)...")
    
    dataset_path = Path("media/pilldata/train")
    images = []
    labels = []
    label_map = {}
    class_idx = 0
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_img_files = []
    for ext in image_extensions:
        all_img_files.extend(list(dataset_path.glob(ext)))
    
    # Group by pill class
    pill_groups = {}
    for img_path in all_img_files:
        pill_name = extract_pill_name(img_path.stem)
        if not pill_name:
            continue
        if pill_name not in pill_groups:
            pill_groups[pill_name] = []
        pill_groups[pill_name].append(img_path)
    
    # Load samples (skip first 50 if training was done on those)
    for pill_name in sorted(pill_groups.keys()):
        label_map[pill_name] = class_idx
        class_idx += 1
        
        # Use images at position [50:50+num_per_class] for test
        for img_path in pill_groups[pill_name][50:50+num_per_class]:
            try:
                img = load_img(str(img_path), target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label_map[pill_name])
            except Exception as e:
                pass
    
    logger.info(f"✓ Loaded {len(images)} test images from {len(label_map)} classes")
    
    X = np.array(images)
    y = np.array(labels)
    y_onehot = to_categorical(y, num_classes=len(label_map))
    
    return X, y, y_onehot, label_map

def run_diagnostic():
    """Run diagnostic on model"""
    logger.info("=" * 80)
    logger.info("PILL CLASSIFIER MODEL DIAGNOSTIC")
    logger.info("=" * 80 + "\n")
    
    # Check if model exists
    model_path = Path("media/pilldata/model_feature_learning.keras")
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    # Load model
    logger.info("Loading model...")
    model = tf.keras.models.load_model(str(model_path))
    logger.info("✓ Model loaded")
    
    # Load metadata
    metadata_path = Path("media/pilldata/model_feature_learning_metadata.json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            logger.info(f"\n📊 Model Info:")
            logger.info(f"  Classes: {metadata.get('num_classes', 'N/A')}")
            logger.info(f"  Model type: {metadata.get('model_type', 'N/A')}")
            logger.info(f"  Training samples: {metadata.get('training_samples', 'N/A')}")
    
    # Load test data
    logger.info("\n📷 Preparing test data...")
    X_test, y_test, y_onehot, label_map = load_test_data(num_per_class=20)
    
    # Get class names
    class_names = sorted(label_map.items(), key=lambda x: x[1])
    class_names = [name for name, _ in class_names]
    
    # Evaluate
    logger.info("\n🔍 Evaluating model...")
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    overall_accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"\n📈 RESULTS:")
    logger.info(f"  Overall Accuracy: {overall_accuracy:.2%}")
    logger.info(f"  Test Samples: {len(X_test)}")
    
    # Per-class accuracy
    logger.info(f"\n📊 Per-Class Accuracy:")
    per_class_acc = {}
    for class_idx in range(len(class_names)):
        mask = y_test == class_idx
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            per_class_acc[class_names[class_idx]] = acc
            status = "✓" if acc >= 0.7 else "⚠" if acc >= 0.5 else "✗"
            logger.info(f"  {status} {class_names[class_idx]:30s} {acc:.2%}")
    
    # Confidence analysis
    logger.info(f"\n💯 Confidence Analysis:")
    max_confidences = np.max(y_pred_probs, axis=1)
    logger.info(f"  Mean confidence: {max_confidences.mean():.2%}")
    logger.info(f"  Min confidence:  {max_confidences.min():.2%}")
    logger.info(f"  Max confidence:  {max_confidences.max():.2%}")
    
    low_confidence = (max_confidences < 0.6).sum()
    logger.info(f"  Low confidence (<60%): {low_confidence}/{len(X_test)} ({low_confidence/len(X_test):.1%})")
    
    # Safety assessment
    logger.info(f"\n🛡️ SAFETY ASSESSMENT:")
    
    problems = []
    
    # Check accuracy
    if overall_accuracy < 0.5:
        problems.append("Low overall accuracy (<50%)")
    elif overall_accuracy < 0.7:
        problems.append("Moderate accuracy (50-70%)")
    
    # Check per-class issues
    problem_classes = [name for name, acc in per_class_acc.items() if acc < 0.5]
    if problem_classes:
        problems.append(f"{len(problem_classes)} classes with <50% accuracy")
    
    # Check confidence
    if low_confidence / len(X_test) > 0.3:
        problems.append("High number of low-confidence predictions")
    
    if not problems:
        verdict = "✅ SAFE FOR DEPLOYMENT"
        logger.info(f"  {verdict}")
        logger.info(f"  Model shows good performance across all classes")
    elif len(problems) <= 2:
        verdict = "⚠️ CAUTION - Use with confidence thresholds"
        logger.info(f"  {verdict}")
        for p in problems:
            logger.info(f"    - {p}")
    else:
        verdict = "❌ RETRAIN NEEDED"
        logger.info(f"  {verdict}")
        for p in problems:
            logger.info(f"    - {p}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    try:
        run_diagnostic()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
