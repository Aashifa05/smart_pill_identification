#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RUN_MODEL_DIAGNOSTIC.py
=======================

Quick runner to analyze your existing pill classifier models.
Execute this to get a safety assessment.

Usage:
    python run_model_diagnostic.py --model model_95.keras
    python run_model_diagnostic.py --model model_anti_overfit.keras
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Silence TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow import keras
    print("✓ TensorFlow loaded")
except ImportError:
    print("❌ ERROR: TensorFlow not installed")
    print("Install with: pip install tensorflow==2.14.0")
    sys.exit(1)

try:
    import cv2
    print("✓ OpenCV loaded")
except ImportError:
    print("❌ ERROR: OpenCV not installed")
    print("Install with: pip install opencv-python==4.8.0.76")
    sys.exit(1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


class QuickModelDiagnostic:
    """Fast diagnostic for existing models"""
    
    def __init__(self, model_path: str, metadata_path: str = None):
        """Load model and metadata"""
        self.model_path = model_path
        self.metadata_path = metadata_path
        
        # Load model
        print(f"\n📦 Loading model: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            sys.exit(1)
        
        # Load metadata if available
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✓ Metadata loaded")
            except:
                print(f"⚠ Could not load metadata: {metadata_path}")
    
    def load_test_images(self, test_dir: str, limit: int = 500):
        """Load test images from directory"""
        print(f"\n📷 Loading test images from: {test_dir}")
        
        images = []
        labels = []
        class_names = []
        
        if not os.path.exists(test_dir):
            print(f"❌ ERROR: Test directory not found: {test_dir}")
            return None, None, None
        
        # Get class directories
        classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        
        if not classes:
            print(f"❌ ERROR: No class directories found in {test_dir}")
            return None, None, None
        
        print(f"Found {len(classes)} classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
        
        count = 0
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(test_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_file in image_files[:100]:  # Max 100 per class
                try:
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Resize to model input size (224x224)
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.0  # Normalize
                    
                    images.append(img)
                    labels.append(class_idx)
                    count += 1
                    
                    if count >= limit:
                        break
                except Exception as e:
                    continue
            
            if count >= limit:
                break
            
            class_names.append(class_name)
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"✓ Loaded {len(images)} test images from {len(set(labels))} classes")
        
        return images, labels, classes
    
    def assess_model_quality(self, test_images, test_labels):
        """Quick quality assessment"""
        print("\n🔍 DIAGNOSTIC ASSESSMENT")
        print("=" * 70)
        
        # Get predictions
        print("Making predictions...")
        predictions = self.model.predict(test_images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        # Overall metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"  Overall Accuracy:      {accuracy:.1%}")
        print(f"  Average Confidence:    {avg_confidence:.1%}")
        print(f"  Confidence Std Dev:    {std_confidence:.3f}")
        print(f"  Min Confidence:        {min_confidence:.1%}")
        print(f"  Max Confidence:        {max_confidence:.1%}")
        
        # Per-class metrics
        print(f"\n📈 PER-CLASS BREAKDOWN:")
        print(f"  {'Class':<20} {'Accuracy':<12} {'Precision':<12} {'Confidence':<12} {'Status':<12}")
        print(f"  {'-'*70}")
        
        problem_classes = []
        per_class_metrics = {}
        
        for class_idx in range(len(set(test_labels))):
            mask = test_labels == class_idx
            if not np.any(mask):
                continue
            
            class_accuracy = accuracy_score(test_labels[mask], predicted_labels[mask])
            class_confidence = np.mean(confidences[mask])
            
            try:
                class_precision = precision_score(test_labels[mask], predicted_labels[mask], zero_division=0)
            except:
                class_precision = 0
            
            # Determine status
            if class_accuracy < 0.70:
                status = "🔴 CRITICAL"
                problem_classes.append((class_idx, class_accuracy, "Low accuracy"))
            elif class_accuracy < 0.80:
                status = "🟡 WARNING"
                problem_classes.append((class_idx, class_accuracy, "Below 80%"))
            elif class_confidence < 0.70:
                status = "🟡 UNCERTAIN"
                problem_classes.append((class_idx, class_accuracy, "Low confidence"))
            else:
                status = "🟢 OK"
            
            per_class_metrics[class_idx] = {
                'accuracy': class_accuracy,
                'confidence': class_confidence,
                'precision': class_precision
            }
            
            print(f"  {class_idx:<20} {class_accuracy:<12.1%} {class_precision:<12.1%} {class_confidence:<12.1%} {status:<12}")
        
        # Assess safety
        print(f"\n🛡️  SAFETY ASSESSMENT:")
        
        critical_count = sum(1 for _, acc, _ in problem_classes if acc < 0.70)
        warning_count = len(problem_classes) - critical_count
        
        if accuracy >= 0.85 and avg_confidence >= 0.75 and critical_count == 0:
            safety_level = "PASS ✅"
            safety_recommendation = "SAFE TO USE"
        elif accuracy >= 0.75 and avg_confidence >= 0.65 and critical_count == 0:
            safety_level = "CONDITIONAL ⚠️"
            safety_recommendation = "USE WITH CAUTION"
        else:
            safety_level = "FAIL ❌"
            safety_recommendation = "NEEDS RETRAINING"
        
        print(f"  Safety Level:          {safety_level}")
        print(f"  Recommendation:        {safety_recommendation}")
        print(f"  Critical Issues:       {critical_count}")
        print(f"  Warnings:              {warning_count}")
        
        # Problem classes
        if problem_classes:
            print(f"\n⚠️  PROBLEM CLASSES:")
            for class_idx, accuracy, reason in sorted(problem_classes, key=lambda x: x[1])[:10]:
                print(f"  Class {class_idx:<3}: {accuracy:<6.1%} accuracy - {reason}")
        
        # Confidence analysis
        print(f"\n📊 CONFIDENCE ANALYSIS:")
        print(f"  Predictions with >80% confidence: {np.sum(confidences > 0.80)}/{len(confidences)} ({np.mean(confidences > 0.80):.1%})")
        print(f"  Predictions with 50-80% confidence: {np.sum((confidences > 0.50) & (confidences <= 0.80))}/{len(confidences)}")
        print(f"  Predictions with <50% confidence: {np.sum(confidences <= 0.50)}/{len(confidences)}")
        
        # Final decision
        print(f"\n{'='*70}")
        print(f"🎯 FINAL VERDICT:")
        print(f"{'='*70}")
        
        if safety_recommendation == "SAFE TO USE":
            print(f"✅ Model is ready for production use")
            print(f"   - Deploy with standard monitoring")
        elif safety_recommendation == "USE WITH CAUTION":
            print(f"⚠️  Model can be used with safeguards:")
            print(f"   - Set confidence threshold to 0.80")
            print(f"   - Implement human review for <0.80 confidence")
            print(f"   - Monitor accuracy weekly")
            print(f"   - Plan retraining in 1 month")
        else:  # NEEDS RETRAINING
            print(f"❌ Model needs retraining before production use")
            print(f"   - Root causes: {critical_count} critical issues")
            print(f"   - Plan comprehensive retraining effort")
            print(f"   - Consider ensemble approaches")
            print(f"   - Increase training data diversity")
        
        # Return assessment
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'safety_level': safety_level,
            'recommendation': safety_recommendation,
            'critical_issues': critical_count,
            'warnings': warning_count,
            'problem_classes': problem_classes,
            'per_class_metrics': per_class_metrics
        }


def main():
    parser = argparse.ArgumentParser(description='Quick diagnostic for pill classifier models')
    parser.add_argument('--model', default='model_95.keras', help='Model file to test')
    parser.add_argument('--metadata', default=None, help='Metadata file for model')
    parser.add_argument('--test-dir', default='media/pilldata/test', help='Test data directory')
    parser.add_argument('--limit', type=int, default=500, help='Max images to test')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = f'media/pilldata/{args.model}'
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found: {model_path}")
        print(f"\nAvailable models:")
        for f in os.listdir('media/pilldata'):
            if f.endswith('.keras'):
                print(f"  - {f}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("PILL CLASSIFIER MODEL DIAGNOSTIC")
    print("="*70)
    print(f"Model:     {args.model}")
    print(f"Test Dir:  {args.test_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Run diagnostic
    diag = QuickModelDiagnostic(model_path, args.metadata)
    test_images, test_labels, class_names = diag.load_test_images(args.test_dir, args.limit)
    
    if test_images is None:
        print("❌ Could not load test data")
        sys.exit(1)
    
    assessment = diag.assess_model_quality(test_images, test_labels)
    
    # Save results
    os.makedirs('diagnostic_results', exist_ok=True)
    
    results = {
        'model': args.model,
        'timestamp': datetime.now().isoformat(),
        'assessment': {
            'accuracy': float(assessment['accuracy']),
            'avg_confidence': float(assessment['avg_confidence']),
            'safety_level': assessment['safety_level'],
            'recommendation': assessment['recommendation'],
            'critical_findings': assessment['critical_issues'],
            'warnings': assessment['warnings'],
            'requirements': len([p for p in assessment['problem_classes'] if p[1] < 0.80]),
            'risk_score': int((1 - assessment['accuracy']) * 100)
        }
    }
    
    with open('diagnostic_results/diagnostic_assessment.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: diagnostic_results/diagnostic_assessment.json")
    
    # Generate decision report
    print("\n📄 Generating safety decision report...")
    os.system('python model_safety_decision.py')


if __name__ == "__main__":
    main()
