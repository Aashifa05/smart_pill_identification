#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QUICK_START_DIAGNOSTIC.md
=========================

How to evaluate if your pill classifier model is safe for medical use.
"""

# Quick Start: Run Diagnostic

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║          PILL CLASSIFIER - MODEL SAFETY EVALUATION (5 MINUTES)             ║
╚════════════════════════════════════════════════════════════════════════════╝

STEP 1: RUN THE DIAGNOSTIC
═══════════════════════════════════════════════════════════════════════════════

Execute this command in your terminal:

    python run_model_diagnostic.py --model model_95.keras

Or test another model:

    python run_model_diagnostic.py --model model_anti_overfit.keras

WHAT IT DOES:
  ✓ Loads your trained model
  ✓ Tests it on your test dataset
  ✓ Calculates per-class accuracy
  ✓ Analyzes confidence scores
  ✓ Identifies problem classes
  ✓ Generates safety assessment
  
EXPECTED RUNTIME: 2-5 minutes (depending on test data size)

═══════════════════════════════════════════════════════════════════════════════

STEP 2: UNDERSTAND THE RESULTS
═══════════════════════════════════════════════════════════════════════════════

The diagnostic will show 3 key sections:

📊 OVERALL METRICS
──────────────────
  Overall Accuracy:       How often model gets right answer (target: >80%)
  Average Confidence:     How sure model is of predictions (target: >70%)
  Confidence Std Dev:     How consistent confidence is
  Min/Max Confidence:     Range of confidence scores

📈 PER-CLASS BREAKDOWN
──────────────────────
  Shows each pill class individually:
  
    Class 0 | 85% | 82% | 78% | 🟢 OK
    Class 1 | 72% | 70% | 65% | 🟡 WARNING
    Class 2 | 55% | 50% | 40% | 🔴 CRITICAL
    
  Status Legend:
    🟢 OK       = Good accuracy (>80%) & good confidence (>70%)
    🟡 WARNING  = Accuracy 70-80% or confidence 65-70%
    🔴 CRITICAL = Accuracy <70% (potentially dangerous)

🛡️ SAFETY ASSESSMENT
────────────────────
  Safety Level:     PASS ✅ / CONDITIONAL ⚠️ / FAIL ❌
  Recommendation:   SAFE TO USE / USE WITH CAUTION / NEEDS RETRAINING

═══════════════════════════════════════════════════════════════════════════════

STEP 3: INTERPRET YOUR VERDICT
═══════════════════════════════════════════════════════════════════════════════

❌ FAIL / NEEDS RETRAINING
───────────────────────────
Symptoms:
  • Overall accuracy < 75%
  • Average confidence < 65%
  • 3+ critical issues (red flags)

What to do:
  ❌ DO NOT deploy to production
  ✓ Run training with train_feature_learning.py
  ✓ Take 1-2 weeks for retraining
  ✓ Run diagnostic again to confirm improvement
  
Command:
  python train_feature_learning.py --epochs 50 --batch-size 32


⚠️ CONDITIONAL / USE WITH CAUTION
──────────────────────────────────
Symptoms:
  • Overall accuracy 75-85%
  • Average confidence 65-75%
  • 1-2 critical issues (but not severe)

What to do:
  ✓ Can deploy but with safeguards
  ✓ Must implement confidence thresholds:
    - Require 0.80+ confidence for identified pills
    - Return "Unknown Tablet" for <0.80 confidence
  ✓ Require human review for uncertain cases
  ✓ Log all predictions for monitoring
  ✓ Plan retraining in 1 month

Configuration:
  confidence_threshold = 0.80  # Reject below this
  review_threshold = 0.70     # Human review 0.70-0.80
  unknown_threshold = 0.50    # "Unknown" below this


✅ PASS / SAFE TO USE
────────────────────
Symptoms:
  • Overall accuracy > 85%
  • Average confidence > 75%
  • 0 critical issues
  • Most classes showing "🟢 OK"

What to do:
  ✓ Model is safe for production
  ✓ Deploy with standard monitoring
  ✓ Continue logging for quality assurance
  ✓ Retrain quarterly with new data

═══════════════════════════════════════════════════════════════════════════════

KEY THRESHOLDS FOR MEDICAL SAFETY
═══════════════════════════════════════════════════════════════════════════════

For Production Use:
  ✓ Overall Accuracy:      Must be ≥ 80%
  ✓ Average Confidence:    Must be ≥ 70%
  ✓ Critical Issues:       Must be ≤ 1
  ✓ Warning Classes:       Acceptable if < 20%

For Per-Class Performance:
  ✓ Problem Class Threshold:  < 70% accuracy = CRITICAL
  ✓ Warning Class Threshold:  70-80% accuracy = CAUTION
  ✓ Safe Class Threshold:     > 80% accuracy = OK

Confidence Thresholds for Real Use:
  ✓ High Confidence:     > 0.80  → Accept prediction, no review
  ✓ Medium Confidence:   0.50-0.80 → Accept with human review
  ✓ Low Confidence:      < 0.50  → Return "Unknown Tablet"

═══════════════════════════════════════════════════════════════════════════════

EXAMPLE RESULTS & INTERPRETATION
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Model is Safe
─────────────────────────
  Overall Accuracy:      87%  ✓ Above 85%
  Average Confidence:    81%  ✓ Above 75%
  Critical Issues:       0    ✓ None
  
  VERDICT: ✅ SAFE TO USE
  ACTION: Deploy immediately

EXAMPLE 2: Model Needs Caution
──────────────────────────────
  Overall Accuracy:      78%  ⚠ Below 85% but above 75%
  Average Confidence:    68%  ⚠ Below 75% but above 65%
  Critical Issues:       1    ⚠ One class has <70% accuracy
  
  VERDICT: ⚠️ USE WITH CAUTION
  ACTION: Deploy with safeguards, confidence threshold 0.80

EXAMPLE 3: Model Needs Retraining
──────────────────────────────────
  Overall Accuracy:      62%  ❌ Below 75%
  Average Confidence:    54%  ❌ Below 65%
  Critical Issues:       8    ❌ Many classes with <70% accuracy
  
  VERDICT: ❌ NEEDS RETRAINING
  ACTION: Stop, retrain for 1-2 weeks, then re-evaluate

═══════════════════════════════════════════════════════════════════════════════

NEXT STEPS
═══════════════════════════════════════════════════════════════════════════════

IF SAFE ✅:
  1. Deploy to production
  2. Set up monitoring (log all predictions)
  3. Track accuracy weekly
  4. Retrain quarterly with new data

IF CONDITIONAL ⚠️:
  1. Set confidence threshold to 0.80 in code
  2. Implement human review workflow
  3. Deploy with confidence checks enabled
  4. Monitor daily for first month
  5. Retrain in 1 month (30 days)

IF NEEDS RETRAINING ❌:
  1. DON'T DEPLOY - critical issues
  2. Run: python train_feature_learning.py
  3. Wait 40-60 minutes for training
  4. Run diagnostic again: python run_model_diagnostic.py
  5. Only deploy when status changes to PASS or CONDITIONAL

═══════════════════════════════════════════════════════════════════════════════

GETTING HELP
═══════════════════════════════════════════════════════════════════════════════

Medical Safety Questions?
  → See: README_MEDICAL_SAFE.md

Technical Details?
  → See: IMPLEMENTATION_GUIDE.md

Quick Reference?
  → See: QUICK_REFERENCE_MEDICAL_SAFE.md

Need to Improve Model?
  → See: train_feature_learning.py

═══════════════════════════════════════════════════════════════════════════════
""")

# Save this text as markdown
import os

guide_text = """
# QUICK START: Model Safety Evaluation

## 5-Minute Diagnostic

Run this to determine if your model is safe for production:

```bash
python run_model_diagnostic.py --model model_95.keras
```

## Interpreting Results

### ✅ PASS / SAFE TO USE
- Overall accuracy > 85%
- Average confidence > 75%
- 0 critical issues
- **Action**: Deploy immediately

### ⚠️ CONDITIONAL / USE WITH CAUTION
- Overall accuracy 75-85%
- Average confidence 65-75%
- 1-2 manageable issues
- **Action**: Deploy with confidence thresholds (0.80+) and human review

### ❌ FAIL / NEEDS RETRAINING
- Overall accuracy < 75%
- Average confidence < 65%
- Multiple critical issues
- **Action**: Stop, retrain for 1-2 weeks, then re-evaluate

## Key Medical Safety Thresholds

| Metric | Threshold | Status |
|--------|-----------|--------|
| Overall Accuracy | > 85% | SAFE ✅ |
| | 75-85% | CAUTION ⚠️ |
| | < 75% | RETRAIN ❌ |
| Average Confidence | > 75% | SAFE ✅ |
| | 65-75% | CAUTION ⚠️ |
| | < 65% | RETRAIN ❌ |
| Critical Classes | 0 | SAFE ✅ |
| | 1-2 | CAUTION ⚠️ |
| | 3+ | RETRAIN ❌ |

## Next Steps

1. **Run diagnostic**: `python run_model_diagnostic.py`
2. **Check results**: Look for verdict in output
3. **Based on verdict**:
   - SAFE: Deploy immediately
   - CAUTION: Deploy with safeguards
   - FAIL: Retrain with `python train_feature_learning.py`
4. **If retraining needed**: Wait 40-60 minutes, then run diagnostic again

## Safety Thresholds for Predictions

When using the model in production:

```python
confidence = model.predict(pill_image)

if confidence > 0.80:
    # Safe to use - high confidence
    return pill_name
elif confidence > 0.50:
    # Uncertain - human review required
    return "REQUIRES_HUMAN_REVIEW"
else:
    # Too uncertain - return unknown
    return "UNKNOWN_TABLET"
```

---

**Questions?** See README_MEDICAL_SAFE.md for full documentation
"""

os.makedirs('diagnostic_results', exist_ok=True)
with open('diagnostic_results/QUICK_START_DIAGNOSTIC.md', 'w') as f:
    f.write(guide_text)

print("\n✓ Guide saved to: diagnostic_results/QUICK_START_DIAGNOSTIC.md")
