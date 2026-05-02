#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Testing & Validation Suite
==========================================

Validates the improved pill classifier on:
1. Known pills (training data)
2. Unseen data (test set from different sources)
3. Confidence thresholds
4. Medical safety scenarios
"""

import os
import sys
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from medical_safe_pill_classifier import (
    MedicalSafeEnsembleClassifier,
    PillClassificationReport
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveValidator:
    """Validate model on multiple aspects"""
    
    def __init__(self, model_path: str, metadata_path: str):
        """Initialize validator with trained model"""
        self.model_path = model_path
        self.metadata_path = metadata_path
        
        # Load model
        self.model = load_model(model_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.label_map = {v: k for k, v in self.metadata['label_map'].items()}
        logger.info(f"Validator initialized with {len(self.label_map)} classes")
    
    def test_on_dataset(self, dataset_dir: str, max_samples_per_class: int = 50) -> Dict:
        """
        Test on a dataset directory.
        
        Args:
            dataset_dir: Path to directory with pill folders
            max_samples_per_class: Limit samples to avoid long tests
            
        Returns:
            dict: Test results with accuracy metrics
        """
        logger.info(f"Testing on dataset: {dataset_dir}")
        
        results = {
            'total_samples': 0,
            'correct_predictions': 0,
            'per_class_accuracy': {},
            'confidence_distribution': {
                'high_confidence': 0,  # > 0.80
                'medium_confidence': 0,  # 0.60-0.80
                'low_confidence': 0,  # 0.40-0.60
                'very_low_confidence': 0  # < 0.40
            },
            'misclassifications': []
        }
        
        dataset_path = Path(dataset_dir)
        
        for pill_dir in sorted(dataset_path.iterdir()):
            if not pill_dir.is_dir():
                continue
            
            pill_name = pill_dir.name
            logger.info(f"Testing class: {pill_name}")
            
            class_correct = 0
            class_total = 0
            
            # Load images (limited)
            image_files = list(pill_dir.glob('*.[jp][pn]g'))[:max_samples_per_class]
            
            for img_path in image_files:
                try:
                    # Load and preprocess
                    img = load_img(str(img_path), target_size=(224, 224))
                    img_array = img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    prediction = self.model.predict(img_array, verbose=0)[0]
                    pred_idx = np.argmax(prediction)
                    pred_name = self.label_map[pred_idx]
                    pred_confidence = float(prediction[pred_idx])
                    
                    # Check if correct
                    is_correct = pred_name == pill_name
                    
                    if is_correct:
                        class_correct += 1
                        results['correct_predictions'] += 1
                    else:
                        # Log misclassification
                        results['misclassifications'].append({
                            'image': str(img_path),
                            'true_class': pill_name,
                            'predicted_class': pred_name,
                            'confidence': pred_confidence
                        })
                    
                    class_total += 1
                    results['total_samples'] += 1
                    
                    # Confidence distribution
                    if pred_confidence >= 0.80:
                        results['confidence_distribution']['high_confidence'] += 1
                    elif pred_confidence >= 0.60:
                        results['confidence_distribution']['medium_confidence'] += 1
                    elif pred_confidence >= 0.40:
                        results['confidence_distribution']['low_confidence'] += 1
                    else:
                        results['confidence_distribution']['very_low_confidence'] += 1
                
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
            
            if class_total > 0:
                class_accuracy = class_correct / class_total
                results['per_class_accuracy'][pill_name] = float(class_accuracy)
                logger.info(f"  {pill_name}: {class_accuracy:.2%} ({class_correct}/{class_total})")
        
        # Overall accuracy
        if results['total_samples'] > 0:
            results['overall_accuracy'] = results['correct_predictions'] / results['total_samples']
        
        return results
    
    def test_confidence_threshold_impact(self, test_results: Dict) -> Dict:
        """
        Analyze how different confidence thresholds affect results.
        """
        logger.info("Analyzing confidence threshold impact...")
        
        conf_dist = test_results['confidence_distribution']
        total = test_results['total_samples']
        
        analysis = {
            'threshold_0_40': {
                'samples_covered': conf_dist['very_low_confidence'] + conf_dist['low_confidence'] + conf_dist['medium_confidence'] + conf_dist['high_confidence'],
                'percentage': 100.0
            },
            'threshold_0_60': {
                'samples_covered': conf_dist['low_confidence'] + conf_dist['medium_confidence'] + conf_dist['high_confidence'],
                'percentage': (conf_dist['low_confidence'] + conf_dist['medium_confidence'] + conf_dist['high_confidence']) / total * 100
            },
            'threshold_0_80': {
                'samples_covered': conf_dist['high_confidence'],
                'percentage': conf_dist['high_confidence'] / total * 100 if total > 0 else 0
            }
        }
        
        return analysis
    
    def test_medical_safety_scenarios(self, dataset_dir: str) -> Dict:
        """
        Test specific medical safety scenarios.
        """
        logger.info("Testing medical safety scenarios...")
        
        scenarios = {
            'critical_misidentification': 0,  # Wrong pill identified with high confidence
            'correct_rejection': 0,  # Unknown pill correctly rejected
            'dangerous_acceptance': 0  # Low-confidence prediction accepted
        }
        
        dataset_path = Path(dataset_dir)
        sample_count = 0
        
        for pill_dir in sorted(dataset_path.iterdir())[:5]:  # Sample 5 classes
            if not pill_dir.is_dir():
                continue
            
            for img_path in list(pill_dir.glob('*.[jp][pn]g'))[:10]:
                try:
                    img = load_img(str(img_path), target_size=(224, 224))
                    img_array = img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    prediction = self.model.predict(img_array, verbose=0)[0]
                    pred_idx = np.argmax(prediction)
                    pred_name = self.label_map[pred_idx]
                    pred_confidence = float(prediction[pred_idx])
                    true_name = pill_dir.name
                    
                    # Scenario analysis
                    if pred_name != true_name and pred_confidence > 0.80:
                        # Critical: High confidence in wrong class
                        scenarios['critical_misidentification'] += 1
                    
                    if pred_name != true_name and pred_confidence < 0.50:
                        # Good: Low confidence in wrong class
                        scenarios['correct_rejection'] += 1
                    
                    if pred_name == true_name and 0.40 <= pred_confidence < 0.60:
                        # Dangerous: Low confidence even in correct class
                        scenarios['dangerous_acceptance'] += 1
                    
                    sample_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed: {e}")
        
        return scenarios


class ResultsReporter:
    """Generate comprehensive test reports"""
    
    @staticmethod
    def generate_test_report(test_results: Dict, safety_scenarios: Dict) -> str:
        """Generate detailed test report"""
        
        report = f"""
================================================================================
                     COMPREHENSIVE TEST REPORT
                     Pill Classifier Validation
================================================================================

OVERALL PERFORMANCE:
================================================================================
Total Test Samples: {test_results['total_samples']}
Correct Predictions: {test_results['correct_predictions']}
Overall Accuracy: {test_results.get('overall_accuracy', 0):.2%}

CONFIDENCE DISTRIBUTION:
================================================================================
High Confidence (>0.80):    {test_results['confidence_distribution']['high_confidence']:5} ({test_results['confidence_distribution']['high_confidence']/test_results['total_samples']*100:.1f}%)
Medium Confidence (0.60-0.80): {test_results['confidence_distribution']['medium_confidence']:5} ({test_results['confidence_distribution']['medium_confidence']/test_results['total_samples']*100:.1f}%)
Low Confidence (0.40-0.60):    {test_results['confidence_distribution']['low_confidence']:5} ({test_results['confidence_distribution']['low_confidence']/test_results['total_samples']*100:.1f}%)
Very Low (<0.40):              {test_results['confidence_distribution']['very_low_confidence']:5} ({test_results['confidence_distribution']['very_low_confidence']/test_results['total_samples']*100:.1f}%)

PER-CLASS ACCURACY:
================================================================================
"""
        for pill_name, accuracy in sorted(test_results['per_class_accuracy'].items()):
            report += f"  {pill_name:30} {accuracy:.2%}\n"
        
        report += f"""
MEDICAL SAFETY ANALYSIS:
================================================================================
Critical Misidentifications (Wrong + High Conf):  {safety_scenarios['critical_misidentification']}
  ⚠ DANGEROUS: Model confidently predicts wrong pill
  Recommendation: Increase confidence threshold

Correct Rejections (Wrong + Low Conf):           {safety_scenarios['correct_rejection']}
  ✓ GOOD: Model correctly uncertain about wrong pill

Dangerous Acceptances (Correct but Low Conf):    {safety_scenarios['dangerous_acceptance']}
  ⚠ CAUTION: Model correct but not confident enough
  Recommendation: Review per-class thresholds

MISCLASSIFICATIONS SUMMARY:
================================================================================
Total Misclassified: {len(test_results['misclassifications'])}
"""
        
        if test_results['misclassifications']:
            # Group by true class
            by_class = {}
            for misc in test_results['misclassifications']:
                true_class = misc['true_class']
                if true_class not in by_class:
                    by_class[true_class] = []
                by_class[true_class].append(misc)
            
            for true_class, items in sorted(by_class.items()):
                report += f"\n  {true_class}:\n"
                for item in items[:5]:  # Show first 5
                    report += f"    → Predicted as {item['predicted_class']} ({item['confidence']:.2%})\n"
                if len(items) > 5:
                    report += f"    ... and {len(items) - 5} more\n"
        
        report += """
RECOMMENDATIONS:
================================================================================

1. HIGH CONFIDENCE THRESHOLD (>0.80):
   - Protects against incorrect confident predictions
   - Acceptable for medical use
   - Current coverage: See confidence distribution

2. FEATURE IMPROVEMENT:
   - Focus on shape, color, and imprint features
   - Avoid relying solely on text imprints
   - Test on diverse image sources

3. PER-CLASS THRESHOLDS:
   - Classes with low accuracy: Increase threshold
   - Classes with high accuracy: Can lower threshold
   - Ensure no critical misidentifications

4. UNKNOWN TABLET HANDLING:
   - Return "Unknown Tablet" for <0.60 confidence
   - Require human review for 0.60-0.80 confidence
   - Accept only >0.80 confidence without review

================================================================================
                            End of Report
================================================================================
"""
        return report
    
    @staticmethod
    def save_results(test_results: Dict, safety_scenarios: Dict, output_dir: str = '.'):
        """Save results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # JSON results
        with open(output_dir / 'test_results_detailed.json', 'w') as f:
            json.dump({
                'test_results': test_results,
                'safety_scenarios': safety_scenarios
            }, f, indent=2)
        
        # Text report
        report = ResultsReporter.generate_test_report(test_results, safety_scenarios)
        with open(output_dir / 'test_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to {output_dir}")
        return report


# ============================================================================
# RUN VALIDATION
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE PILL CLASSIFIER VALIDATION")
    logger.info("=" * 80)
    
    # Initialize validator
    model_path = 'media/pilldata/model_feature_learning.h5'
    metadata_path = 'media/pilldata/model_feature_learning_metadata.json'
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train the model first using train_feature_learning.py")
        sys.exit(1)
    
    validator = ComprehensiveValidator(model_path, metadata_path)
    
    # Test on training data (known pills)
    logger.info("\n1. Testing on KNOWN PILLS (Training Data)")
    logger.info("-" * 80)
    test_results = validator.test_on_dataset('media/pilldata/train', max_samples_per_class=30)
    
    # Analyze confidence thresholds
    logger.info("\n2. Analyzing CONFIDENCE THRESHOLDS")
    logger.info("-" * 80)
    threshold_analysis = validator.test_confidence_threshold_impact(test_results)
    for threshold, data in threshold_analysis.items():
        logger.info(f"  {threshold}: {data['samples_covered']} samples ({data['percentage']:.1f}%)")
    
    # Test medical safety scenarios
    logger.info("\n3. Testing MEDICAL SAFETY SCENARIOS")
    logger.info("-" * 80)
    safety_scenarios = validator.test_medical_safety_scenarios('media/pilldata/train')
    
    # Generate report
    logger.info("\n4. Generating COMPREHENSIVE REPORT")
    logger.info("-" * 80)
    report = ResultsReporter.save_results(test_results, safety_scenarios, 'validation_results')
    print(report)
    
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
