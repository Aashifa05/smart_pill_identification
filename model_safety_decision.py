#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Safety Decision Guide
============================

Interprets diagnostic results and provides clear YES/NO decision
on whether model is safe for production use.
"""

import json
from pathlib import Path
from typing import Dict, Tuple


class SafetyDecisionEngine:
    """Determines if model is safe based on diagnostics"""
    
    @staticmethod
    def interpret_safety_level(assessment: Dict) -> Tuple[str, str]:
        """
        Interpret safety level and provide clear guidance.
        
        Returns:
            (decision, reasoning)
        """
        
        safety_level = assessment['safety_level']
        risk_score = assessment['risk_score']
        recommendation = assessment['recommendation']
        
        if safety_level == 'PASS':
            decision = "✅ YES - SAFE TO USE"
            reasoning = """
The model demonstrates acceptable quality for production use.

Confidence: HIGH
Risk Level: LOW
Recommendation: PROCEED WITH DEPLOYMENT

Quality Indicators:
  ✓ Overall accuracy ≥ 80%
  ✓ Average confidence ≥ 70%
  ✓ Low overfitting risk
  ✓ Most classes performing well
  ✓ No critical misclassifications
"""
            
        elif safety_level == 'CONDITIONAL':
            decision = "⚠️  CONDITIONAL - USE WITH CAUTION"
            reasoning = """
The model can be used but requires safety measures and monitoring.

Confidence: MODERATE
Risk Level: MEDIUM
Recommendation: PROCEED WITH SAFEGUARDS

Issues Found:
  ⚠ Some accuracy below 80%
  ⚠ Some confidence below 70%
  ⚠ Possible overfitting detected
  ⚠ Some classes need improvement

REQUIRED SAFEGUARDS:
  1. Implement confidence-based rejection (threshold ≥ 0.80)
  2. Set per-class thresholds based on training performance
  3. Require human review for predictions < 0.80 confidence
  4. Log ALL predictions for monitoring
  5. Weekly accuracy tracking
  6. Plan retraining in 1 month or after 100 predictions

WITHOUT THESE SAFEGUARDS: DO NOT DEPLOY
"""
            
        elif safety_level == 'FAIL':
            decision = "❌ NO - DO NOT USE IN PRODUCTION"
            reasoning = """
The model has significant quality issues and is NOT SAFE for production.

Confidence: LOW
Risk Level: HIGH
Recommendation: NEEDS RETRAINING

Critical Issues:
  ✗ Overall accuracy < 80% (or borderline)
  ✗ Average confidence < 70% (or low)
  ✗ High overfitting risk detected
  ✗ Many classes with poor accuracy
  ✗ Frequent misclassifications
  ✗ Unreliable confidence scores

REQUIRED ACTIONS BEFORE DEPLOYMENT:
  1. DO NOT deploy to production
  2. Analyze root causes:
     - Check training data quality
     - Look for class imbalance
     - Review data distribution
  3. Retrain with improvements:
     - Add more training data
     - Improve data quality
     - Increase augmentation
     - Use longer training
  4. Consider ensemble methods
  5. Re-validate before next attempt

Timeline: 1-2 weeks to retrain and re-evaluate
"""
        
        else:
            decision = "❓ UNKNOWN"
            reasoning = "Unable to determine safety status."
        
        return decision, reasoning
    
    @staticmethod
    def get_action_plan(assessment: Dict) -> str:
        """Get specific action plan"""
        
        plan = ""
        recommendation = assessment['recommendation']
        
        if recommendation == 'SAFE_TO_USE':
            plan = """
IMMEDIATE ACTIONS (Today):
  ☐ Review this diagnostic report
  ☐ Sign off on model quality
  ☐ Prepare deployment checklist

DEPLOYMENT (This week):
  ☐ Package model for production
  ☐ Set up logging/monitoring
  ☐ Deploy to staging first
  ☐ Run integration tests
  ☐ Get medical team approval
  ☐ Deploy to production

ONGOING (After deployment):
  ☐ Monitor predictions daily
  ☐ Track accuracy weekly
  ☐ Review misclassifications monthly
  ☐ Retrain quarterly with new data
  ☐ Update documentation
"""
        
        elif recommendation == 'USE_WITH_CAUTION':
            plan = """
IMMEDIATE ACTIONS (Today):
  ☐ Set up confidence thresholds
  ☐ Implement per-class thresholds
  ☐ Design human review workflow
  ☐ Set up logging system

BEFORE DEPLOYMENT (This week):
  ☐ Test confidence-based rejection
  ☐ Verify human review process
  ☐ Train staff on CAUTION cases
  ☐ Set up monitoring alerts
  ☐ Create runbooks for issues

DEPLOYMENT (Next week):
  ☐ Deploy with safeguards enabled
  ☐ Monitor closely for first week
  ☐ Review human review cases daily
  ☐ Adjust thresholds if needed

ONGOING (After deployment):
  ☐ Daily monitoring and review
  ☐ Weekly accuracy reports
  ☐ Plan retraining (1 month)
  ☐ Adjust thresholds monthly
  ☐ Prepare for retraining
"""
        
        elif recommendation == 'NEEDS_RETRAINING':
            plan = """
IMMEDIATE ACTIONS (Today):
  ☐ DO NOT deploy
  ☐ Document findings
  ☐ Schedule retraining session
  ☐ Analyze problem classes

THIS WEEK:
  ☐ Identify root causes
  ☐ Collect more training data
  ☐ Improve data quality
  ☐ Design improved training approach
  ☐ Plan model architecture changes

NEXT WEEK:
  ☐ Retrain model with improvements
  ☐ Run full diagnostic again
  ☐ Compare with previous results
  ☐ Prepare for revalidation

BEFORE DEPLOYMENT:
  ☐ Achieve target accuracy (>80%)
  ☐ Verify confidence scores
  ☐ Pass medical review
  ☐ Test in staging environment
"""
        
        else:  # DO_NOT_USE_PRODUCTION
            plan = """
IMMEDIATE ACTIONS (TODAY):
  ☐ DO NOT DEPLOY - CRITICAL ISSUES
  ☐ Notify stakeholders
  ☐ Document critical findings
  ☐ Schedule emergency meeting

THIS WEEK:
  ☐ Perform root cause analysis
  ☐ Identify fundamental issues
  ☐ Decide on model architecture changes
  ☐ Plan major retraining effort
  ☐ Identify new data sources

NEXT 2 WEEKS:
  ☐ Collect and prepare new training data
  ☐ Consider ensemble approaches
  ☐ Redesign training pipeline
  ☐ Plan extended training
  ☐ Prepare for comprehensive revalidation

BEFORE ANY DEPLOYMENT:
  ☐ Achieve >85% accuracy
  ☐ Verify robust confidence scores
  ☐ Extensive independent validation
  ☐ Medical team approval
  ☐ Extended monitoring plan
"""
        
        return plan
    
    @staticmethod
    def generate_decision_summary(assessment: Dict) -> str:
        """Generate one-page decision summary"""
        
        decision, reasoning = SafetyDecisionEngine.interpret_safety_level(assessment)
        action_plan = SafetyDecisionEngine.get_action_plan(assessment)
        
        summary = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    MODEL SAFETY DECISION SUMMARY                          ║
║                   Medical Use - Production Readiness                       ║
╚════════════════════════════════════════════════════════════════════════════╝

DECISION:
════════════════════════════════════════════════════════════════════════════════
{decision}

════════════════════════════════════════════════════════════════════════════════
INTERPRETATION:
════════════════════════════════════════════════════════════════════════════════
{reasoning}

════════════════════════════════════════════════════════════════════════════════
RISK ASSESSMENT:
════════════════════════════════════════════════════════════════════════════════

Overall Risk Score:        {assessment['risk_score']:.1f}/100
Safety Level:              {assessment['safety_level']}

Critical Findings:         {len(assessment['critical_findings'])}
Warnings:                  {len(assessment['warnings'])}
Requirements:              {len(assessment['requirements'])}

════════════════════════════════════════════════════════════════════════════════
ACTION PLAN:
════════════════════════════════════════════════════════════════════════════════
{action_plan}

════════════════════════════════════════════════════════════════════════════════
SIGN-OFF:
════════════════════════════════════════════════════════════════════════════════

Medical Safety Officer: ________________  Date: __________

Technical Lead:        ________________  Date: __________

Project Manager:       ________________  Date: __________

════════════════════════════════════════════════════════════════════════════════
"""
        
        return summary


def generate_decision_report():
    """Main function to generate decision report"""
    
    # Load diagnostic assessment
    diag_file = Path('diagnostic_results/diagnostic_assessment.json')
    
    if not diag_file.exists():
        print("❌ ERROR: diagnostic_assessment.json not found")
        print("Please run: python model_diagnostic_analyzer.py")
        return
    
    with open(diag_file, 'r') as f:
        data = json.load(f)
    
    assessment = data['assessment']
    
    # Generate decision
    engine = SafetyDecisionEngine()
    summary = engine.generate_decision_summary(assessment)
    
    # Print and save
    print(summary)
    
    with open('diagnostic_results/SAFETY_DECISION.txt', 'w') as f:
        f.write(summary)
    
    print("\n✓ Safety decision saved to: diagnostic_results/SAFETY_DECISION.txt")


if __name__ == "__main__":
    generate_decision_report()
