#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Reference: Training for Generalization
=============================================

Usage examples and best practices for training models that
generalize well to unseen real-world pill images.
"""

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║        TRAINING PILL CLASSIFIER FOR GENERALIZATION                   ║
║          (Avoiding Overfitting on Unseen Real-World Images)          ║
╚═══════════════════════════════════════════════════════════════════════╝

🎯 GOAL
───────
Train a model that:
  ✓ Learns general pill features (shape, color, size, texture)
  ✓ Works on unseen pill images
  ✓ Tolerates lighting variations, angles, image quality
  ✓ Doesn't memorize training data
  ✓ Provides reliable real-world performance

═════════════════════════════════════════════════════════════════════════
TRAINING SCRIPT: train_generalization_focused.py
═════════════════════════════════════════════════════════════════════════

✅ WHAT IT DOES:
───────────────

1. DATA AUGMENTATION
   • Rotation (±20°)          → Learn rotation invariance
   • Zoom (±20%)              → Learn scale invariance
   • Shift (±15% H/V)         → Learn position invariance
   • Brightness (±20%)        → Learn lighting robustness
   • Color shift              → Learn color robustness
   • Horizontal flip          → Learn symmetry

2. REGULARIZATION
   • L1/L2 regularization     → Penalize weight complexity
   • Dropout (40%, 30%)       → Prevent co-adaptation
   • Batch normalization      → Stabilize learning
   • Class weights            → Balance imbalanced data

3. TRAINING STRATEGY
   • Early stopping           → Stop overfitting early
   • Learning rate scheduling → Better convergence
   • Proper data splits       → True generalization measurement
   • Transfer learning        → Leverage pre-trained features

4. MONITORING
   • Generalization metrics   → Watch train vs val gap
   • Overfitting detection    → Alert if gap > 5%
   • Test set evaluation      → Unseen data performance
   • Visualization plots      → See learning curves

═════════════════════════════════════════════════════════════════════════
SETUP
═════════════════════════════════════════════════════════════════════════

1. ORGANIZE YOUR DATA:
   ───────────────────
   
   Create this folder structure:
   
   media/pilldata/pills/
   ├── Aspirin/
   │   ├── aspirin_1.jpg
   │   ├── aspirin_2.jpg
   │   └── aspirin_3.jpg
   ├── Ibuprofen/
   │   ├── ibuprofen_1.jpg
   │   ├── ibuprofen_2.jpg
   │   └── ibuprofen_3.jpg
   ├── Acetaminophen/
   │   └── ...
   └── ...other pill types...
   
   Requirements:
   • Minimum 50 images per class
   • Recommended 100-300 images per class
   • Supported formats: .jpg, .png
   • Any size (will be resized to 224×224)

2. INSTALL DEPENDENCIES:
   ──────────────────────
   
   pip install tensorflow>=2.14.0
   pip install keras>=2.14.0
   pip install opencv-python
   pip install numpy pandas scikit-learn matplotlib

═════════════════════════════════════════════════════════════════════════
QUICK START (3 STEPS)
═════════════════════════════════════════════════════════════════════════

STEP 1: Organize data (see SETUP above)
───────

STEP 2: Run training script
───────

  cd c:\\Users\\THINKBOOK\\OneDrive\\Desktop\\Admin
  
  python train_generalization_focused.py

STEP 3: Monitor results
────────
  
  Check the output:
  • generalization_metrics.png  ← Visualization
  • model_generalization.keras  ← Trained model
  • model_generalization_metadata.json ← Configuration

═════════════════════════════════════════════════════════════════════════
EXPECTED TRAINING OUTPUT
═════════════════════════════════════════════════════════════════════════

Loading pill data from media/pilldata/pills...
  Loading class 1/10: Aspirin (150 images)
  Loading class 2/10: Ibuprofen (140 images)
  ✓ Loaded 1420 images from 10 classes

Data split:
  Training:   1019 images (71.7%)
  Validation: 213 images (15.0%)
  Test:       188 images (13.2%)

Building generalization-focused model...
✓ Model built with 2,856,842 parameters

TRAINING

Epoch 1/100:   100/32 ▁▂▃▄▅▆▇█ - 45s 142ms/step
Train Loss: 2.3421, Val Loss: 2.4521
Train Acc=0.2345, Val Acc=0.2104, Overfit Score=0.0241

Epoch 2/100:   100/32 ▁▂▃▄▅▆▇█ - 42s 131ms/step
Train Loss: 1.8234, Val Loss: 1.8765
Train Acc=0.4521, Val Acc=0.4312, Overfit Score=0.0209

...

Epoch 32/100:  100/32 ▁▂▃▄▅▆▇█ - 40s 125ms/step
Train Loss: 0.1234, Val Loss: 0.1567
Train Acc=0.9234, Val Acc=0.9087, Overfit Score=0.0147 ✓ Good!

...stopping training (validation loss stopped improving)...

EVALUATION ON TEST SET (UNSEEN DATA)

Test Accuracy: 0.8830 = 88.30%
Test Loss: 0.3521

Detailed Test Metrics:
  Precision: 0.8801
  Recall:    0.8795
  F1 Score:  0.8798

✓ EXCELLENT: Model generalizes well to unseen data!

✓ Model saved to media/pilldata/model_generalization.keras
✓ Metadata saved to media/pilldata/model_generalization_metadata.json
✓ Generalization plot saved to media/pilldata/generalization_metrics.png

═════════════════════════════════════════════════════════════════════════
UNDERSTANDING THE OUTPUT
═════════════════════════════════════════════════════════════════════════

KEY METRICS:
────────────

1. Train Acc vs Val Acc:
   
   Train Acc 0.92, Val Acc 0.90
   Gap: 2% ✓ EXCELLENT (generalizing well)
   
   Train Acc 0.95, Val Acc 0.70
   Gap: 25% ✗ OVERFITTING (memorizing data)

2. Overfitting Score:
   
   Score: 0.02 (2%)  ✓ Perfect
   Score: 0.05 (5%)  ✓ Good
   Score: 0.10 (10%) ⚠ Mild overfitting
   Score: 0.20+ ✗ Severe overfitting

3. Test Accuracy:
   
   Test > Val: ✓ Great! Model improving on unseen data
   Test ≈ Val: ✓ Good! Consistent generalization
   Test < Val: ⚠ Warning! Possible different data distribution

GENERALIZATION METRICS PLOT:
────────────────────────────

The script creates 4 visualizations:

┌─────────────┬──────────────┐
│ Loss Curve  │ Accuracy     │
│ (should     │ Curve        │
│ converge &  │ (should      │
│ stay close) │ stay close)  │
├─────────────┼──────────────┤
│ Overfitting │ Train-Val    │
│ Score       │ Gap          │
│ (should     │ (should be   │
│ stay ~0)    │ green)       │
└─────────────┴──────────────┘

═════════════════════════════════════════════════════════════════════════
USING THE TRAINED MODEL
═════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Basic Prediction
─────────────────────────────

from Users.utility.multi_feature_pill_classifier import MultiFeaturePillClassifier

classifier = MultiFeaturePillClassifier(
    'media/pilldata/model_generalization.keras',
    'media/pilldata/model_generalization_metadata.json'
)

result = classifier.predict('pill_image.jpg')
print(result['pill_name'])
print(result['confidence'])

EXAMPLE 2: Batch Processing
──────────────────────────────

from pathlib import Path

classifier = MultiFeaturePillClassifier(...)

image_dir = Path('images/pills')
for image_path in image_dir.glob('*.jpg'):
    result = classifier.predict(str(image_path))
    print(f"{image_path.name}: {result['pill_name']}")

EXAMPLE 3: Integration with Django
───────────────────────────────────

from django.http import JsonResponse
from Users.utility.multi_feature_pill_classifier import MultiFeaturePillClassifier

def classify_pill(request):
    if request.method == 'POST':
        image = request.FILES['image']
        
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            for chunk in image.chunks():
                tmp.write(chunk)
            tmp.flush()
            
            classifier = MultiFeaturePillClassifier(...)
            result = classifier.predict(tmp.name)
        
        return JsonResponse(result)

═════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═════════════════════════════════════════════════════════════════════════

PROBLEM: "No data found"
SOLUTION:
  • Check folder structure: media/pilldata/pills/ClassName/images.jpg
  • Use actual class names (not example names)
  • Ensure images are .jpg or .png
  • Check file permissions

PROBLEM: "Training is overfitting (gap > 10%)"
SOLUTION:
  • Increase L1/L2 regularization: L1L2(l1=5e-4, l2=5e-4)
  • Increase dropout: Dropout(0.5) instead of 0.4
  • More augmentation: rotation_range=30, zoom_range=0.3
  • Early stopping will help: it stops before severe overfitting

PROBLEM: "Training is underfitting (both low)"
SOLUTION:
  • Decrease regularization: L1L2(l1=1e-5, l2=1e-5)
  • Decrease dropout: Dropout(0.2)
  • Less augmentation: rotation_range=10, zoom_range=0.1
  • More epochs: epochs=200
  • More training data (if available)

PROBLEM: "Out of memory"
SOLUTION:
  • Reduce batch size: batch_size=16 (instead of 32)
  • Use fewer training epochs
  • Reduce number of classes/images

PROBLEM: "Very slow training"
SOLUTION:
  • This is normal! Model is learning carefully
  • Expected: 30-60 minutes for 1000+ images, 10 classes
  • Running on GPU (if available): much faster
  • Use fewer images for testing: sample smaller dataset

═════════════════════════════════════════════════════════════════════════
ANTI-OVERFITTING TECHNIQUES EXPLAINED
═════════════════════════════════════════════════════════════════════════

1. DATA AUGMENTATION (1000x variations)
   ────────────────────────────────────
   
   Without:  Model sees each image ~100 times
             → Memorizes specific instances
   
   With:     Model sees ~100,000 variations
             → Must learn general patterns
   
   Result: 20-30% improvement on unseen data

2. DROPOUT (Neuron deactivation)
   ─────────────────────────────
   
   Training: 60% of neurons active
             → Each neuron independent
             → No reliance chains
   
   Inference: 100% of neurons active
              → Combine independent features
              → Robust ensemble
   
   Result: Better generalization, less overfitting

3. L1/L2 REGULARIZATION (Weight penalty)
   ──────────────────────────────────────
   
   Loss = Classification Loss + λ × Σ|weights|
                               └─ Penalty for large weights
   
   Result: Simpler models that generalize better

4. BATCH NORMALIZATION (Layer stabilization)
   ──────────────────────────────────────────
   
   Normalizes: y = (x - mean) / std
   
   Effect: Faster convergence, less internal shift
   Result: More stable learning, better generalization

5. EARLY STOPPING (Training stopper)
   ──────────────────────────────────
   
   Monitors val_loss:
   - If improving: continue training
   - If plateau/worsen: stop (overfitting detected)
   
   Result: Catches overfitting at the right time

6. LEARNING RATE SCHEDULING (Adaptive learning)
   ────────────────────────────────────────────
   
   Start with large LR (1e-4): Fast initial learning
   Reduce to small LR (1e-7): Fine-tune details
   
   Result: Better convergence, fewer local minima

═════════════════════════════════════════════════════════════════════════
BEST PRACTICES
═════════════════════════════════════════════════════════════════════════

✅ DO:
───
  • Use data augmentation (always!)
  • Monitor train vs val curves
  • Check for overfitting early
  • Use proper train/val/test splits
  • Balance classes with weights
  • Save best model based on validation
  • Test on completely unseen data
  • Document performance metrics

❌ DON'T:
──────
  • Train on all data (no validation/test)
  • Use same data for training and testing
  • Ignore overfitting signals
  • Train without regularization
  • Use imbalanced classes without weighting
  • Train without early stopping
  • Overfit for higher training accuracy
  • Ignore real-world data distribution

═════════════════════════════════════════════════════════════════════════
EXPECTED PERFORMANCE
═════════════════════════════════════════════════════════════════════════

With 100+ images per class:

Training Accuracy:   90-95%
Validation Accuracy: 88-93%  ← Similar to training
Test Accuracy:       87-92%  ← Good generalization

If you see:
Training: 95%   ✓ Good
Val:      93%   ✓ Good
Test:     91%   ✓ Good
Gap: 2%        ✓ Excellent generalization!

═════════════════════════════════════════════════════════════════════════
RESOURCES
═════════════════════════════════════════════════════════════════════════

Documentation:
  • TRAINING_ANTI_OVERFITTING_GUIDE.md ← Detailed explanation
  • MULTI_FEATURE_CLASSIFICATION_GUIDE.md ← Model usage
  • train_generalization_focused.py ← Source code

Training script: python train_generalization_focused.py
Testing script: python test_multi_feature_classifier.py
Usage examples: python multi_feature_classification_examples.py

═════════════════════════════════════════════════════════════════════════
SUCCESS CHECKLIST
═════════════════════════════════════════════════════════════════════════

Before deployment:
  ☐ Train/Val curves close together
  ☐ Overfitting score < 5%
  ☐ Test accuracy > 85%
  ☐ Model converged (early stopping triggered)
  ☐ Generalization plot looks good
  ☐ All classes have similar accuracy
  ☐ No class has much lower performance
  ☐ Real-world performance validated

═════════════════════════════════════════════════════════════════════════
NEXT STEPS
═════════════════════════════════════════════════════════════════════════

1. Organize your pill images in media/pilldata/pills/
2. Run: python train_generalization_focused.py
3. Check generalization_metrics.png for visual results
4. Verify test accuracy > 85%
5. Use model_generalization.keras for predictions
6. Monitor real-world performance

═════════════════════════════════════════════════════════════════════════

Questions? Check TRAINING_ANTI_OVERFITTING_GUIDE.md for detailed explanations!

Good luck training! Your model will generalize to real-world unseen pills! 🎉
""")
