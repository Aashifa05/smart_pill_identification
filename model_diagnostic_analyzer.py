#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Quality Diagnostic & Safety Assessment
==============================================

Comprehensive evaluation of pill classifier:
1. Identifies misclassified classes
2. Detects confidence issues
3. Assesses generalization problems
4. Provides safety recommendations
5. Generates detailed diagnostic report
"""

import os
import sys
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDiagnosticAnalyzer:
    """Comprehensive model quality assessment"""
    
    def __init__(self, model_path: str, metadata_path: str):
        """Initialize diagnostic analyzer"""
        self.model_path = model_path
        self.metadata_path = metadata_path
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            self.model = None
            self.metadata = None
            return
        
        try:
            # Load model
            self.model = load_model(model_path)
            logger.info(f"✓ Model loaded: {model_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"✓ Metadata loaded: {metadata_path}")
            
            self.label_map = {v: k for k, v in self.metadata['label_map'].items()}
            self.num_classes = len(self.label_map)
            
        except Exception as e:
            logger.error(f"Failed to load model/metadata: {e}")
            self.model = None
            self.metadata = None
    
    def check_model_loaded(self) -> bool:
        """Verify model is loaded"""
        if self.model is None or self.metadata is None:
            logger.error("Model not loaded. Cannot perform diagnostics.")
            return False
        return True
    
    def evaluate_on_dataset(self, dataset_dir: str, 
                          max_samples_per_class: int = 100) -> Dict:
        """
        Evaluate model on dataset and collect detailed metrics
        """
        if not self.check_model_loaded():
            return {}
        
        logger.info(f"\nEVALUATING MODEL ON: {dataset_dir}")
        logger.info("=" * 80)
        
        results = {
            'overall_accuracy': 0,
            'total_samples': 0,
            'correct_predictions': 0,
            'per_class_metrics': {},
            'misclassifications': defaultdict(list),
            'confidence_statistics': {
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'percentiles': {}
            },
            'confusion_matrix': None,
            'problem_classes': []  # Classes with issues
        }
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        dataset_path = Path(dataset_dir)
        
        for pill_dir in sorted(dataset_path.iterdir()):
            if not pill_dir.is_dir():
                continue
            
            pill_name = pill_dir.name
            class_correct = 0
            class_total = 0
            class_confidences = []
            class_misclassifications = defaultdict(int)
            
            # Load images
            image_files = list(pill_dir.glob('*.[jp][pn]g'))[:max_samples_per_class]
            
            if not image_files:
                logger.warning(f"No images found for {pill_name}")
                continue
            
            logger.info(f"\n  Testing: {pill_name:30} ({len(image_files)} images)")
            
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
                    
                    # Get true label index
                    true_idx = self.metadata['label_map'][pill_name]
                    
                    # Check if correct
                    is_correct = pred_name == pill_name
                    
                    if is_correct:
                        class_correct += 1
                        results['correct_predictions'] += 1
                    else:
                        # Track misclassification
                        class_misclassifications[pred_name] += 1
                        results['misclassifications'][pill_name].append({
                            'predicted': pred_name,
                            'confidence': pred_confidence,
                            'image': str(img_path)
                        })
                    
                    class_total += 1
                    results['total_samples'] += 1
                    all_predictions.append(pred_idx)
                    all_targets.append(true_idx)
                    all_confidences.append(pred_confidence)
                    class_confidences.append(pred_confidence)
                
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
            
            if class_total > 0:
                class_accuracy = class_correct / class_total
                
                results['per_class_metrics'][pill_name] = {
                    'accuracy': float(class_accuracy),
                    'correct': class_correct,
                    'total': class_total,
                    'precision': float(class_correct) / class_total,  # Simplified
                    'avg_confidence': float(np.mean(class_confidences)),
                    'min_confidence': float(np.min(class_confidences)),
                    'max_confidence': float(np.max(class_confidences)),
                    'std_confidence': float(np.std(class_confidences)),
                    'misclassification_pattern': dict(class_misclassifications)
                }
                
                # Flag problematic classes
                if class_accuracy < 0.70:
                    results['problem_classes'].append({
                        'name': pill_name,
                        'accuracy': class_accuracy,
                        'reason': 'LOW_ACCURACY'
                    })
                
                if np.mean(class_confidences) < 0.60:
                    results['problem_classes'].append({
                        'name': pill_name,
                        'avg_confidence': float(np.mean(class_confidences)),
                        'reason': 'LOW_CONFIDENCE'
                    })
                
                status = "✓" if class_accuracy >= 0.80 else "⚠" if class_accuracy >= 0.70 else "✗"
                logger.info(f"    {status} Accuracy: {class_accuracy:.2%}, "
                           f"Confidence: {np.mean(class_confidences):.2%} ± {np.std(class_confidences):.2%}")
        
        # Overall metrics
        if results['total_samples'] > 0:
            results['overall_accuracy'] = results['correct_predictions'] / results['total_samples']
        
        # Confidence statistics
        if all_confidences:
            all_confidences = np.array(all_confidences)
            results['confidence_statistics'] = {
                'mean': float(np.mean(all_confidences)),
                'median': float(np.median(all_confidences)),
                'std': float(np.std(all_confidences)),
                'min': float(np.min(all_confidences)),
                'max': float(np.max(all_confidences)),
                'percentiles': {
                    'p25': float(np.percentile(all_confidences, 25)),
                    'p50': float(np.percentile(all_confidences, 50)),
                    'p75': float(np.percentile(all_confidences, 75)),
                    'p90': float(np.percentile(all_confidences, 90)),
                    'p95': float(np.percentile(all_confidences, 95))
                }
            }
        
        # Confusion matrix
        if all_targets and all_predictions:
            results['confusion_matrix'] = confusion_matrix(
                all_targets, all_predictions
            ).tolist()
        
        return results
    
    def analyze_generalization(self, train_results: Dict, 
                              test_results: Dict) -> Dict:
        """Analyze generalization issues"""
        
        logger.info("\n" + "=" * 80)
        logger.info("GENERALIZATION ANALYSIS")
        logger.info("=" * 80)
        
        analysis = {
            'overfitting_risk': 'LOW',
            'overfitting_score': 0.0,
            'issues': [],
            'details': {}
        }
        
        if not train_results or not test_results:
            return analysis
        
        train_acc = train_results.get('overall_accuracy', 0)
        test_acc = test_results.get('overall_accuracy', 0)
        
        # Calculate gap
        gap = train_acc - test_acc
        analysis['train_accuracy'] = train_acc
        analysis['test_accuracy'] = test_acc
        analysis['accuracy_gap'] = gap
        
        # Assess overfitting
        if gap > 0.15:
            analysis['overfitting_risk'] = 'HIGH'
            analysis['overfitting_score'] = min(gap, 0.3)
            analysis['issues'].append('OVERFITTING: Large train-test gap')
        elif gap > 0.10:
            analysis['overfitting_risk'] = 'MODERATE'
            analysis['overfitting_score'] = gap / 2
            analysis['issues'].append('POSSIBLE_OVERFITTING: Noticeable train-test gap')
        else:
            analysis['overfitting_risk'] = 'LOW'
            analysis['overfitting_score'] = 0
        
        # Per-class generalization
        per_class_gaps = {}
        for class_name in test_results.get('per_class_metrics', {}):
            if class_name in train_results.get('per_class_metrics', {}):
                train_class_acc = train_results['per_class_metrics'][class_name]['accuracy']
                test_class_acc = test_results['per_class_metrics'][class_name]['accuracy']
                class_gap = train_class_acc - test_class_acc
                per_class_gaps[class_name] = class_gap
                
                if class_gap > 0.20:
                    analysis['issues'].append(
                        f'CLASS_OVERFITTING: {class_name} (gap={class_gap:.2%})'
                    )
        
        analysis['per_class_gaps'] = per_class_gaps
        
        # Confidence consistency
        train_conf = train_results.get('confidence_statistics', {}).get('mean', 0)
        test_conf = test_results.get('confidence_statistics', {}).get('mean', 0)
        
        if abs(train_conf - test_conf) > 0.15:
            analysis['issues'].append(
                f'CONFIDENCE_INCONSISTENCY: Train={train_conf:.2%}, Test={test_conf:.2%}'
            )
        
        return analysis
    
    def identify_problem_classes(self, results: Dict) -> Dict:
        """Identify classes that need improvement"""
        
        logger.info("\n" + "=" * 80)
        logger.info("PROBLEM CLASS ANALYSIS")
        logger.info("=" * 80)
        
        problems = {
            'low_accuracy_classes': [],
            'low_confidence_classes': [],
            'high_variance_classes': [],
            'commonly_misclassified_as': defaultdict(list),
            'summary': {}
        }
        
        for class_name, metrics in results.get('per_class_metrics', {}).items():
            accuracy = metrics.get('accuracy', 0)
            avg_conf = metrics.get('avg_confidence', 0)
            std_conf = metrics.get('std_confidence', 0)
            
            # Low accuracy
            if accuracy < 0.70:
                problems['low_accuracy_classes'].append({
                    'class': class_name,
                    'accuracy': accuracy,
                    'severity': 'CRITICAL' if accuracy < 0.50 else 'HIGH'
                })
            
            # Low confidence
            if avg_conf < 0.60:
                problems['low_confidence_classes'].append({
                    'class': class_name,
                    'avg_confidence': avg_conf,
                    'severity': 'CRITICAL' if avg_conf < 0.40 else 'HIGH'
                })
            
            # High variance
            if std_conf > 0.25:
                problems['high_variance_classes'].append({
                    'class': class_name,
                    'std_confidence': std_conf,
                    'severity': 'HIGH' if std_conf > 0.35 else 'MEDIUM'
                })
            
            # Misclassification patterns
            for misclass, count in metrics.get('misclassification_pattern', {}).items():
                if count > 2:
                    problems['commonly_misclassified_as'][class_name].append({
                        'misclassified_as': misclass,
                        'count': count
                    })
        
        # Summary
        problems['summary'] = {
            'total_classes': len(results.get('per_class_metrics', {})),
            'classes_with_low_accuracy': len(problems['low_accuracy_classes']),
            'classes_with_low_confidence': len(problems['low_confidence_classes']),
            'classes_with_high_variance': len(problems['high_variance_classes']),
            'total_problem_classes': len(set(
                [p['class'] for p in problems['low_accuracy_classes']] +
                [p['class'] for p in problems['low_confidence_classes']] +
                [p['class'] for p in problems['high_variance_classes']]
            ))
        }
        
        return problems
    
    def assess_safety(self, results: Dict, problems: Dict, 
                     generalization: Dict) -> Dict:
        """Assess if model is safe for production use"""
        
        logger.info("\n" + "=" * 80)
        logger.info("MEDICAL SAFETY ASSESSMENT")
        logger.info("=" * 80)
        
        assessment = {
            'safety_level': 'PASS',
            'risk_score': 0.0,  # 0-100, higher = more risk
            'recommendation': 'SAFE_TO_USE',
            'issues': [],
            'warnings': [],
            'requirements': [],
            'critical_findings': []
        }
        
        # Check overall accuracy
        overall_acc = results.get('overall_accuracy', 0)
        if overall_acc < 0.80:
            assessment['risk_score'] += 30
            assessment['warnings'].append(
                f'Overall accuracy {overall_acc:.2%} is below recommended 0.80'
            )
        if overall_acc < 0.70:
            assessment['risk_score'] += 40
            assessment['critical_findings'].append(
                f'Overall accuracy {overall_acc:.2%} is too low for medical use'
            )
        
        # Check confidence
        conf_mean = results.get('confidence_statistics', {}).get('mean', 0)
        if conf_mean < 0.70:
            assessment['risk_score'] += 25
            assessment['warnings'].append(
                f'Average confidence {conf_mean:.2%} is below recommended 0.70'
            )
        if conf_mean < 0.50:
            assessment['risk_score'] += 35
            assessment['critical_findings'].append(
                f'Average confidence {conf_mean:.2%} is too low - model not reliable'
            )
        
        # Check problem classes
        prob_classes = problems.get('summary', {}).get('total_problem_classes', 0)
        total_classes = problems.get('summary', {}).get('total_classes', 1)
        
        if prob_classes > 0:
            problem_ratio = prob_classes / total_classes
            assessment['risk_score'] += min(problem_ratio * 50, 30)
            assessment['warnings'].append(
                f'{prob_classes}/{total_classes} classes have issues'
            )
        
        # Check generalization
        if generalization.get('overfitting_risk') == 'HIGH':
            assessment['risk_score'] += 25
            assessment['warnings'].append('HIGH overfitting risk detected')
        if generalization.get('overfitting_risk') == 'MODERATE':
            assessment['risk_score'] += 15
            assessment['warnings'].append('Moderate overfitting risk detected')
        
        # Check critical misclassifications
        if problems.get('commonly_misclassified_as'):
            critical_misclass = sum(
                1 for class_name, misclass_list in problems['commonly_misclassified_as'].items()
                if any(m['count'] > 5 for m in misclass_list)
            )
            if critical_misclass > 0:
                assessment['risk_score'] += 20
                assessment['critical_findings'].append(
                    f'{critical_misclass} classes frequently misclassified'
                )
        
        # Cap risk score at 100
        assessment['risk_score'] = min(assessment['risk_score'], 100)
        
        # Determine safety level and recommendation
        if assessment['critical_findings']:
            assessment['safety_level'] = 'FAIL'
            assessment['recommendation'] = 'DO_NOT_USE_PRODUCTION'
        elif assessment['risk_score'] >= 70:
            assessment['safety_level'] = 'FAIL'
            assessment['recommendation'] = 'NEEDS_RETRAINING'
        elif assessment['risk_score'] >= 50:
            assessment['safety_level'] = 'CONDITIONAL'
            assessment['recommendation'] = 'USE_WITH_CAUTION'
            assessment['requirements'].append('Implement human review for <0.80 confidence')
            assessment['requirements'].append('Set per-class confidence thresholds')
            assessment['requirements'].append('Monitor predictions for misclassification')
        else:
            assessment['safety_level'] = 'PASS'
            assessment['recommendation'] = 'SAFE_TO_USE'
            assessment['requirements'].append('Implement standard monitoring')
            assessment['requirements'].append('Monthly accuracy tracking')
            assessment['requirements'].append('Retrain quarterly with new data')
        
        return assessment


class DiagnosticReportGenerator:
    """Generate comprehensive diagnostic report"""
    
    @staticmethod
    def generate_report(model_name: str, results: Dict, 
                       problems: Dict, generalization: Dict,
                       assessment: Dict) -> str:
        """Generate detailed diagnostic report"""
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║            PILL CLASSIFIER - MODEL QUALITY DIAGNOSTIC REPORT              ║
║                     Medical Safety Assessment                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Model: {model_name}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

══════════════════════════════════════════════════════════════════════════════
                          EXECUTIVE SUMMARY
══════════════════════════════════════════════════════════════════════════════

Safety Level:        {assessment['safety_level']}
Risk Score:          {assessment['risk_score']:.1f}/100
Recommendation:      {assessment['recommendation']}

Overall Accuracy:    {results.get('overall_accuracy', 0):.2%}
Total Samples:       {results.get('total_samples', 0)}
Correct Predictions: {results.get('correct_predictions', 0)}

Avg Confidence:      {results.get('confidence_statistics', {}).get('mean', 0):.2%}
Confidence Std:      {results.get('confidence_statistics', {}).get('std', 0):.2%}
Confidence Range:    {results.get('confidence_statistics', {}).get('min', 0):.2%} - {results.get('confidence_statistics', {}).get('max', 0):.2%}

══════════════════════════════════════════════════════════════════════════════
                      CRITICAL FINDINGS
══════════════════════════════════════════════════════════════════════════════

"""
        
        if assessment['critical_findings']:
            for finding in assessment['critical_findings']:
                report += f"🔴 {finding}\n"
        else:
            report += "✓ No critical issues found\n"
        
        report += f"""
══════════════════════════════════════════════════════════════════════════════
                      PER-CLASS PERFORMANCE
══════════════════════════════════════════════════════════════════════════════

"""
        
        # Sort by accuracy
        sorted_classes = sorted(
            results.get('per_class_metrics', {}).items(),
            key=lambda x: x[1]['accuracy']
        )
        
        for class_name, metrics in sorted_classes:
            accuracy = metrics['accuracy']
            avg_conf = metrics['avg_confidence']
            
            if accuracy >= 0.80:
                status = "✓"
            elif accuracy >= 0.70:
                status = "⚠"
            else:
                status = "✗"
            
            report += f"{status} {class_name:30} Acc: {accuracy:6.2%}  Conf: {avg_conf:6.2%}\n"
        
        report += f"""
══════════════════════════════════════════════════════════════════════════════
                    PROBLEM CLASSES REQUIRING ATTENTION
══════════════════════════════════════════════════════════════════════════════

Low Accuracy Classes (<0.70):
"""
        for p in problems['low_accuracy_classes']:
            report += f"  • {p['class']:30} {p['accuracy']:.2%} [SEVERITY: {p['severity']}]\n"
        
        report += f"""
Low Confidence Classes (<0.60):
"""
        for p in problems['low_confidence_classes']:
            report += f"  • {p['class']:30} {p['avg_confidence']:.2%} [SEVERITY: {p['severity']}]\n"
        
        report += f"""
High Variance Classes (std >0.25):
"""
        for p in problems['high_variance_classes']:
            report += f"  • {p['class']:30} σ={p['std_confidence']:.2%} [SEVERITY: {p['severity']}]\n"
        
        report += f"""
Commonly Misclassified Pairs:
"""
        for class_name, misclass_list in problems['commonly_misclassified_as'].items():
            if misclass_list:
                report += f"  • {class_name} → {misclass_list[0]['misclassified_as']} ({misclass_list[0]['count']}x)\n"
        
        report += f"""
══════════════════════════════════════════════════════════════════════════════
                      GENERALIZATION ASSESSMENT
══════════════════════════════════════════════════════════════════════════════

Overfitting Risk:    {generalization.get('overfitting_risk', 'UNKNOWN')}
Overfitting Score:   {generalization.get('overfitting_score', 0):.2%}

Train Accuracy:      {generalization.get('train_accuracy', 0):.2%}
Test Accuracy:       {generalization.get('test_accuracy', 0):.2%}
Accuracy Gap:        {generalization.get('accuracy_gap', 0):.2%}

"""
        if generalization.get('issues'):
            for issue in generalization['issues']:
                report += f"⚠ {issue}\n"
        else:
            report += "✓ No major generalization issues\n"
        
        report += f"""
══════════════════════════════════════════════════════════════════════════════
                      SAFETY REQUIREMENTS
══════════════════════════════════════════════════════════════════════════════

"""
        if assessment['requirements']:
            for req in assessment['requirements']:
                report += f"  • {req}\n"
        else:
            report += "  ✓ Standard monitoring sufficient\n"
        
        report += f"""
══════════════════════════════════════════════════════════════════════════════
                      RECOMMENDATIONS
══════════════════════════════════════════════════════════════════════════════

"""
        
        if assessment['recommendation'] == 'SAFE_TO_USE':
            report += """
✓ Model is SAFE for production use with standard monitoring.

Recommended Actions:
  1. Deploy to production
  2. Implement prediction logging
  3. Monitor accuracy monthly
  4. Retrain quarterly with new data
  5. Keep human review as backup
"""
        
        elif assessment['recommendation'] == 'USE_WITH_CAUTION':
            report += """
⚠ Model can be used but requires additional safety measures.

Required Actions:
  1. Implement confidence-based rejection (threshold: 0.80)
  2. Set per-class confidence thresholds
  3. Require human review for predictions <0.80 confidence
  4. Log all predictions for analysis
  5. Increase monitoring frequency (weekly)
  6. Schedule retraining after 100 predictions or 1 month
  
Do NOT use without these safeguards.
"""
        
        elif assessment['recommendation'] == 'NEEDS_RETRAINING':
            report += """
✗ Model NEEDS RETRAINING before production use.

Required Actions:
  1. Analyze misclassification patterns
  2. Add more training data for problematic classes
  3. Retrain with improved augmentation
  4. Increase training epochs
  5. Use ensemble methods if available
  6. Re-evaluate before deployment
"""
        
        else:  # DO_NOT_USE_PRODUCTION
            report += """
✗✗✗ DO NOT USE IN PRODUCTION ✗✗✗

This model is NOT SAFE for medical use. Critical issues detected:
"""
            for finding in assessment['critical_findings']:
                report += f"  • {finding}\n"
            
            report += """
Required Actions:
  1. Do NOT deploy to production
  2. Perform root cause analysis
  3. Collect more training data
  4. Improve data quality
  5. Try different model architectures
  6. Use ensemble methods
  7. Implement extensive validation before retry
"""
        
        report += f"""
══════════════════════════════════════════════════════════════════════════════
                          END OF REPORT
══════════════════════════════════════════════════════════════════════════════

For detailed analysis, see associated JSON files and confusion matrices.
"""
        
        return report
    
    @staticmethod
    def save_report(report: str, output_path: str):
        """Save report to file"""
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"✓ Report saved to {output_path}")


import pandas as pd

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("PILL CLASSIFIER - MODEL QUALITY DIAGNOSTIC")
    logger.info("=" * 80)
    
    # Find available models
    model_paths = [
        ('Feature Learning', 'media/pilldata/model_feature_learning.h5', 
         'media/pilldata/model_feature_learning_metadata.json'),
        ('Enhanced', 'media/pilldata/model_enhanced.keras',
         'media/pilldata/model_enhanced_metadata.json'),
        ('Anti-Overfit', 'media/pilldata/model_anti_overfit.keras',
         'media/pilldata/model_metadata.json'),
        ('Standard', 'media/pilldata/model.keras',
         'media/pilldata/model_metadata.json'),
    ]
    
    # Find first existing model
    model_to_test = None
    for name, model_path, meta_path in model_paths:
        if os.path.exists(model_path) and os.path.exists(meta_path):
            model_to_test = (name, model_path, meta_path)
            break
    
    if not model_to_test:
        logger.error("No trained models found!")
        logger.info("\nTo train a model, run:")
        logger.info("  python train_feature_learning.py")
        sys.exit(1)
    
    name, model_path, meta_path = model_to_test
    logger.info(f"\n✓ Found model: {name}")
    
    # Initialize analyzer
    analyzer = ModelDiagnosticAnalyzer(model_path, meta_path)
    
    if not analyzer.check_model_loaded():
        sys.exit(1)
    
    # Evaluate on training data
    logger.info("\n[PHASE 1] Evaluating on TRAINING data...")
    train_results = analyzer.evaluate_on_dataset('media/pilldata/train', max_samples_per_class=50)
    
    # Evaluate on test data (if available)
    logger.info("\n[PHASE 2] Evaluating on TEST data...")
    test_results = {}
    if os.path.exists('media/pilldata/test'):
        test_results = analyzer.evaluate_on_dataset('media/pilldata/test', max_samples_per_class=50)
    else:
        logger.warning("Test directory not found, using training data for evaluation")
        test_results = train_results
    
    # Analyze generalization
    logger.info("\n[PHASE 3] Analyzing generalization...")
    generalization = analyzer.analyze_generalization(train_results, test_results)
    
    # Identify problem classes
    logger.info("\n[PHASE 4] Identifying problem classes...")
    problems = analyzer.identify_problem_classes(test_results)
    
    # Assess safety
    logger.info("\n[PHASE 5] Assessing medical safety...")
    assessment = analyzer.assess_safety(test_results, problems, generalization)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Safety Level:       {assessment['safety_level']}")
    logger.info(f"Risk Score:         {assessment['risk_score']:.1f}/100")
    logger.info(f"Recommendation:     {assessment['recommendation']}")
    logger.info(f"Overall Accuracy:   {test_results.get('overall_accuracy', 0):.2%}")
    logger.info(f"Avg Confidence:     {test_results.get('confidence_statistics', {}).get('mean', 0):.2%}")
    
    # Generate report
    logger.info("\n[PHASE 6] Generating detailed report...")
    report = DiagnosticReportGenerator.generate_report(
        name, test_results, problems, generalization, assessment
    )
    
    # Print and save
    print(report)
    
    # Save JSON results
    output_dir = Path('diagnostic_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'diagnostic_assessment.json', 'w') as f:
        json.dump({
            'assessment': assessment,
            'results': {
                'overall_accuracy': test_results.get('overall_accuracy', 0),
                'total_samples': test_results.get('total_samples', 0),
                'confidence_statistics': test_results.get('confidence_statistics', {})
            },
            'problems': {
                'summary': problems.get('summary', {}),
                'low_accuracy_count': len(problems.get('low_accuracy_classes', [])),
                'low_confidence_count': len(problems.get('low_confidence_classes', []))
            },
            'generalization': {
                'overfitting_risk': generalization.get('overfitting_risk', ''),
                'train_test_gap': generalization.get('accuracy_gap', 0)
            }
        }, f, indent=2)
    
    # Save text report
    DiagnosticReportGenerator.save_report(report, output_dir / 'diagnostic_report.txt')
    
    # Save detailed metrics
    with open(output_dir / 'per_class_metrics.json', 'w') as f:
        json.dump(test_results.get('per_class_metrics', {}), f, indent=2)
    
    logger.info(f"\n✓ Results saved to diagnostic_results/")
    logger.info(f"✓ Diagnostic complete!")
