#!/usr/bin/env python
"""
Medical-Safe Pill Classifier - Implementation Checklist
========================================================

Follow this checklist to implement the improved classifier in your system.
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import logging
logger = logging.getLogger(__name__)


class ImplementationChecklist:
    """Track implementation progress"""
    
    def __init__(self):
        self.items = {
            'PHASE_1_SETUP': [
                ('Check Python Version', False, '3.8+'),
                ('Install Required Packages', False, 'tensorflow, keras, opencv-python'),
                ('Verify Data Structure', False, 'media/pilldata/train/ populated'),
                ('Check Model Path', False, 'media/pilldata/ accessible'),
            ],
            'PHASE_2_TRAINING': [
                ('Run Training Script', False, 'python train_feature_learning.py'),
                ('Verify Model Created', False, 'model_feature_learning.h5 exists'),
                ('Check Metadata Generated', False, 'model_feature_learning_metadata.json exists'),
                ('Review Training Plots', False, 'training_history.png shows convergence'),
            ],
            'PHASE_3_VALIDATION': [
                ('Run Validation Tests', False, 'python validate_classifier.py'),
                ('Review Test Results', False, 'Check per-class accuracy > 80%'),
                ('Analyze Confidence Distribution', False, 'Most predictions > 0.80'),
                ('Check Medical Safety Scenarios', False, 'No critical misidentifications'),
            ],
            'PHASE_4_INTEGRATION': [
                ('Test Classifier Directly', False, 'python medical_safe_pill_classifier.py'),
                ('Test Django Integration', False, 'python pill_integration.py test_image.jpg'),
                ('Verify Ensemble Setup', False, 'All models loading correctly'),
                ('Check Output Format', False, 'Predictions include all fields'),
            ],
            'PHASE_5_DEPLOYMENT': [
                ('Configure Confidence Thresholds', False, '0.80 for medical safety'),
                ('Set Per-Class Thresholds', False, 'Based on validation results'),
                ('Implement Human Review Process', False, 'For CAUTION predictions'),
                ('Setup Audit Logging', False, 'All predictions logged'),
                ('Create Fallback Procedure', False, 'When pill cannot be identified'),
            ],
            'PHASE_6_MONITORING': [
                ('Monitor Prediction Accuracy', False, 'Track over time'),
                ('Review Misclassifications', False, 'Monthly analysis'),
                ('Retrain Schedule', False, 'Quarterly with new data'),
                ('Update Documentation', False, 'As system evolves'),
            ]
        }
    
    def print_checklist(self):
        """Print interactive checklist"""
        print("\n" + "=" * 80)
        print("IMPLEMENTATION CHECKLIST - Medical-Safe Pill Classifier")
        print("=" * 80 + "\n")
        
        for phase_name, items in self.items.items():
            phase_num = phase_name.split('_')[1]
            print(f"\n🔵 PHASE {phase_num}: {phase_name.replace('PHASE_', '').replace('_', ' ')}")
            print("-" * 80)
            
            for idx, (task, completed, details) in enumerate(items, 1):
                status = "✓" if completed else "○"
                print(f"  {status} {idx}. {task}")
                print(f"     Details: {details}\n")
    
    def run_automated_checks(self):
        """Run automated verification checks"""
        print("\n" + "=" * 80)
        print("AUTOMATED SETUP VERIFICATION")
        print("=" * 80 + "\n")
        
        checks = {
            'Python TensorFlow': self._check_tensorflow(),
            'Model Files': self._check_model_files(),
            'Data Structure': self._check_data_structure(),
            'Required Scripts': self._check_scripts(),
            'Imports': self._check_imports(),
        }
        
        print("\nRESULTS:")
        print("-" * 80)
        
        all_passed = True
        for check_name, (passed, message) in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status:10} {check_name:30} {message}")
            all_passed = all_passed and passed
        
        print("\n" + "=" * 80)
        if all_passed:
            print("✓ All checks PASSED - Ready to proceed")
        else:
            print("✗ Some checks FAILED - See details above")
        print("=" * 80 + "\n")
        
        return all_passed
    
    def _check_tensorflow(self):
        """Check TensorFlow installation"""
        try:
            import tensorflow as tf
            version = tf.__version__
            if version >= '2.10':
                return True, f"v{version} installed"
            else:
                return False, f"v{version} (need >=2.10)"
        except ImportError:
            return False, "TensorFlow not installed"
    
    def _check_model_files(self):
        """Check for existing model files"""
        models = [
            'media/pilldata/model_feature_learning.h5',
            'media/pilldata/model.keras',
            'media/pilldata/model_enhanced.keras',
        ]
        
        existing = [m for m in models if os.path.exists(m)]
        
        if existing:
            return True, f"{len(existing)} model(s) found"
        else:
            return False, "No models found (train with train_feature_learning.py)"
    
    def _check_data_structure(self):
        """Check dataset structure"""
        train_dir = Path('media/pilldata/train')
        
        if not train_dir.exists():
            return False, "media/pilldata/train/ not found"
        
        pill_dirs = list(train_dir.iterdir())
        
        if not pill_dirs:
            return False, "No pill classes in train/"
        
        total_images = sum(len(list(d.glob('*.[jp][pn]g'))) for d in pill_dirs if d.is_dir())
        
        if total_images < 10:
            return False, f"Only {total_images} images (need more training data)"
        
        return True, f"{len(pill_dirs)} classes, {total_images} images"
    
    def _check_scripts(self):
        """Check required scripts exist"""
        scripts = [
            'train_feature_learning.py',
            'validate_classifier.py',
            'medical_safe_pill_classifier.py',
            'pill_integration.py',
        ]
        
        existing = [s for s in scripts if os.path.exists(s)]
        missing = [s for s in scripts if s not in existing]
        
        if missing:
            return False, f"Missing: {', '.join(missing)}"
        else:
            return True, "All scripts present"
    
    def _check_imports(self):
        """Check key imports"""
        try:
            from medical_safe_pill_classifier import MedicalSafeEnsembleClassifier
            from pill_integration import PillIdentificationService
            return True, "All imports successful"
        except ImportError as e:
            return False, f"Import failed: {str(e)}"


class QuickStart:
    """Quick start guide"""
    
    @staticmethod
    def print_guide():
        """Print quick start guide"""
        guide = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                 MEDICAL-SAFE PILL CLASSIFIER - QUICK START                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

THREE SIMPLE STEPS TO BETTER PILL CLASSIFICATION
==================================================

STEP 1️⃣  TRAIN THE MODEL (40-60 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Command:
    python train_feature_learning.py

  What it does:
    ✓ Loads all pills from media/pilldata/train/
    ✓ Creates augmented versions (brightness, rotation, blur)
    ✓ Trains deep learning model with feature learning
    ✓ Saves model_feature_learning.h5
    ✓ Generates training_history.png

  Expected output:
    Test Accuracy: 0.85+
    Per-class accuracy > 80% for most pills


STEP 2️⃣  VALIDATE & TEST (5 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Command:
    python validate_classifier.py

  What it checks:
    ✓ Accuracy on known pills (training data)
    ✓ Confidence distribution (should be mostly >80%)
    ✓ Medical safety scenarios
    ✓ Potential misclassifications

  Output files:
    validation_results/test_report.txt
    validation_results/test_results_detailed.json


STEP 3️⃣  INTEGRATE INTO YOUR APP (10 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Option A: Use the service directly
  ──────────────────────────────────

    from pill_integration import PillIdentificationService

    service = PillIdentificationService()
    result = service.identify_pill('pill_image.jpg')

    if result['risk_level'] == 'SAFE':
        print(f"✓ Pill identified: {result['tablet_name']}")
    elif result['risk_level'] == 'CAUTION':
        print(f"⚠ Requires review: {result['tablet_name']}")
    else:
        print(f"✗ Cannot identify: {result['tablet_name']}")


  Option B: Use in Django view
  ────────────────────────────

    from pill_integration import PillIdentificationService

    class PillIdentificationView(View):
        def __init__(self):
            self.service = PillIdentificationService()

        def post(self, request):
            image = request.FILES['image']
            result = self.service.identify_pill(image.path)
            return JsonResponse(result)


  Option C: Advanced features
  ──────────────────────────

    from medical_safe_pill_classifier import (
        MedicalSafeEnsembleClassifier,
        PillClassificationReport
    )

    # Initialize with multiple models
    classifier = MedicalSafeEnsembleClassifier(
        model_paths=[...],
        metadata_path='...',
        confidence_threshold=0.80
    )

    # Predict
    prediction = classifier.predict('pill.jpg')

    # Generate report
    report = PillClassificationReport.generate_prediction_report(
        prediction, 'pill.jpg'
    )
    print(report)


MEDICAL SAFETY RULES
════════════════════════════════════════════════════════════════════════════════

Status          Risk Level    Confidence    Action
────────────────────────────────────────────────────────────────────────────────
IDENTIFIED      SAFE          >0.80         ✓ Use immediately
UNCERTAIN       CAUTION       0.60-0.80     ⚠ Require human review
UNKNOWN         REJECT        <0.60         ✗ Cannot identify

KEY FEATURES
════════════════════════════════════════════════════════════════════════════════

1. LEARNS ACTUAL PILL FEATURES
   ✓ Shape (aspect ratio, roundness)
   ✓ Color (RGB, saturation, brightness)
   ✓ Imprint (presence, contrast)
   ✓ Texture (edges, fine details)
   
   NOT just reading text imprints!

2. HIGH CONFIDENCE REQUIREMENTS
   ✓ Minimum 80% for identified pills
   ✓ Per-class thresholds available
   ✓ Rejects uncertain predictions

3. UNKNOWN PILL HANDLING
   ✓ Returns "Unknown Tablet" for low confidence
   ✓ Prevents forcing wrong predictions
   ✓ Medical-safe default

4. ENSEMBLE VOTING
   ✓ Combines multiple trained models
   ✓ More robust than single model
   ✓ Reduces individual model errors

5. DETAILED REPORTS
   ✓ Medical-grade safety information
   ✓ Feature extraction details
   ✓ Risk level assessment


EXPECTED IMPROVEMENTS
════════════════════════════════════════════════════════════════════════════════

Problem                          Before      After       Improvement
────────────────────────────────────────────────────────────────────────────────
Misclassification (trained data)  High        Low         Ensemble + Features
Performance (unseen data)         Poor        Good        Robust features
Confidence handling              Forced      Safe         Thresholds
Unknown pill detection           No          Yes         Rejection mechanism

EXAMPLE OUTPUTS
════════════════════════════════════════════════════════════════════════════════

Good Prediction:
{
    "status": "IDENTIFIED",
    "tablet_name": "Aspirin 500mg",
    "confidence": 0.94,
    "risk_level": "SAFE",
    "reason": "High confidence match (94%)"
}

⚠️  Uncertain Prediction:
{
    "status": "UNCERTAIN",
    "tablet_name": "Ibuprofen 200mg",
    "confidence": 0.68,
    "risk_level": "CAUTION",
    "reason": "Moderate confidence (68%) - human review needed"
}

✗ Cannot Identify:
{
    "status": "UNKNOWN",
    "tablet_name": "Unknown Tablet",
    "confidence": 0.32,
    "risk_level": "REJECT",
    "reason": "No confident match found"
}

TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════════

❌ Low accuracy after training?
   → Check if training images are clear
   → Add more diverse images (different angles, lighting)
   → Train for more epochs (increase from 50 to 75)

❌ Too many "Unknown Tablets"?
   → Lower threshold slightly (0.75 instead of 0.80)
   → Check if pill class is in training data
   → Add more training images for problematic classes

❌ Model too confident on wrong pills?
   → Increase threshold (0.85 instead of 0.80)
   → Ensemble voting helps - make sure all models loaded
   → Review training data quality

NEXT STEPS
════════════════════════════════════════════════════════════════════════════════

1. Run the training: python train_feature_learning.py
2. Validate results: python validate_classifier.py
3. Test on your images: python pill_integration.py your_pill.jpg
4. Review README_MEDICAL_SAFE.md for detailed documentation
5. Integrate into your Django application
6. Set up human review process for CAUTION predictions
7. Monitor predictions and retrain monthly

SUPPORT
════════════════════════════════════════════════════════════════════════════════

For detailed info:      see README_MEDICAL_SAFE.md
For API reference:      see docstrings in medical_safe_pill_classifier.py
For Django integration: see pill_integration.py

================================================================================
                    Ready to start? Run: python train_feature_learning.py
================================================================================
"""
        print(guide)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Medical-Safe Pill Classifier - Implementation Checklist'
    )
    parser.add_argument('--check', action='store_true', 
                       help='Run automated checks')
    parser.add_argument('--checklist', action='store_true',
                       help='Show implementation checklist')
    parser.add_argument('--quickstart', action='store_true',
                       help='Show quick start guide')
    
    args = parser.parse_args()
    
    if not args.check and not args.checklist and not args.quickstart:
        # Default: show everything
        args.check = True
        args.checklist = True
        args.quickstart = True
    
    if args.check:
        print("\n")
        checklist = ImplementationChecklist()
        checklist.run_automated_checks()
    
    if args.checklist:
        checklist = ImplementationChecklist()
        checklist.print_checklist()
    
    if args.quickstart:
        QuickStart.print_guide()
