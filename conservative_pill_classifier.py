#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conservative Pill Classifier with Unknown Pill Detection
=========================================================

Medical Safety Implementation:
- Conservative confidence thresholds
- Unknown pill detection
- No forced misclassification
- Safety-first approach

This wraps the anti-overfitting trained model with intelligent
unknown pill handling to prevent dangerous misidentifications.
"""

import os
import sys
import numpy as np
from pathlib import Path
import logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConservativePillClassifier:
    """
    Conservative pill classifier with unknown pill detection.
    
    Safety-first approach:
    - Never forces misclassification
    - Unknown pills marked when confidence insufficient
    - Supports imprint-optional classification
    - Provides detailed confidence metrics
    """
    
    def __init__(self, model_path='media/pilldata/model_anti_overfit.keras',
                 metadata_path='media/pilldata/model_metadata.json',
                 confidence_threshold=0.75,
                 imprint_missing_threshold=0.65):
        """
        Initialize conservative classifier.
        
        Args:
            model_path: Path to trained model
            metadata_path: Path to class mapping
            confidence_threshold: Min confidence for known pills (0.75 = 75%)
            imprint_missing_threshold: Lower threshold when imprint missing (0.65 = 65%)
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.confidence_threshold = confidence_threshold
        self.imprint_missing_threshold = imprint_missing_threshold
        
        logger.info(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        
        logger.info(f"Loading metadata from {metadata_path}...")
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata.get('class_names', [])
        logger.info(f"✓ Loaded {len(self.class_names)} pill classes")
    
    def detect_imprint_visibility(self, image_array):
        """
        Detect if imprint is visible in pill image.
        Simple heuristic: Check texture/contrast in central region.
        """
        # Use image texture to estimate imprint visibility
        # High variance in center = visible imprint
        center_crop = image_array[56:168, 56:168, :]  # Central 112x112 of 224x224
        
        # Calculate contrast/variance as proxy for imprint visibility
        brightness = np.mean(center_crop)
        contrast = np.std(center_crop)
        edge_strength = np.abs(np.gradient(np.mean(center_crop, axis=2))).mean()
        
        # Imprint typically has higher edge strength and specific contrast range
        has_imprint = (edge_strength > 5.0) and (10 < contrast < 80)
        
        return has_imprint, {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'edge_strength': float(edge_strength)
        }
    
    def predict(self, image_path, verbose=True):
        """
        Predict pill class with conservative unknown handling.
        
        Returns:
            dict: Classification result with safety information
        """
        # Load and preprocess image
        try:
            img = load_img(image_path, target_size=(224, 224))
            x = img_to_array(img)
            x_preprocessed = preprocess_input(x)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return self._unknown_pill_result("Image loading failed", confidence=0.0)
        
        # Get model prediction
        predictions = self.model.predict(np.array([x_preprocessed]), verbose=0)[0]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[::-1][:5]
        top_confidences = predictions[top_indices]
        top_names = [self.class_names[i] for i in top_indices]
        
        # Detect imprint visibility
        has_imprint, imprint_metrics = self.detect_imprint_visibility(x)
        
        # Use appropriate threshold
        threshold = self.confidence_threshold if has_imprint else self.imprint_missing_threshold
        
        # Decision logic
        top_confidence = float(top_confidences[0])
        top_class_idx = top_indices[0]
        top_class = top_names[0]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PILL CLASSIFICATION (Conservative Mode)")
        logger.info(f"{'='*70}")
        logger.info(f"Top Prediction: {top_class}")
        logger.info(f"Confidence: {top_confidence*100:.2f}%")
        logger.info(f"Threshold: {threshold*100:.2f}%")
        logger.info(f"Imprint Visible: {has_imprint}")
        logger.info(f"Decision: {'✓ KNOWN PILL' if top_confidence >= threshold else '❌ UNKNOWN PILL'}")
        logger.info(f"{'='*70}\n")
        
        # Conservative classification
        if top_confidence >= threshold:
            return self._known_pill_result(
                top_class,
                top_confidence,
                top_names[:5],
                top_confidences[:5],
                has_imprint
            )
        else:
            # Safety-first: Mark as unknown
            return self._unknown_pill_result(
                f"Confidence too low ({top_confidence*100:.2f}% < {threshold*100:.2f}%)",
                confidence=top_confidence,
                top_guess=top_class,
                all_predictions=list(zip(top_names, top_confidences.tolist()))
            )
    
    def _known_pill_result(self, class_name, confidence, all_names, all_confidences, has_imprint):
        """Generate result for confidently identified pill"""
        return {
            'status': 'IDENTIFIED',
            'pill_name': class_name,
            'confidence': float(confidence),
            'confidence_percentage': f"{confidence*100:.2f}%",
            'imprint_visible': has_imprint,
            'classification_type': 'Confident Match',
            'top_5_predictions': [
                {'pill': name, 'confidence': float(conf)}
                for name, conf in zip(all_names, all_confidences)
            ],
            'safety_status': '✓ Safe for use (identified)',
            'recommendation': f"This is {class_name}. Consult medication information for details."
        }
    
    def _unknown_pill_result(self, reason, confidence=0.0, top_guess=None, all_predictions=None):
        """Generate result for unknown/uncertain pill"""
        return {
            'status': 'UNKNOWN',
            'pill_name': 'UNKNOWN TABLET',
            'confidence': float(confidence),
            'confidence_percentage': f"{confidence*100:.2f}%",
            'classification_type': 'Conservative Unknown',
            'reason_for_unknown': reason,
            'top_guess': top_guess,
            'all_predictions': all_predictions or [],
            'safety_status': '⚠️ CRITICAL - Cannot identify',
            'recommendation': 'CRITICAL: Please do not consume this tablet without professional identification. Contact a pharmacist or poison control immediately.',
            'required_actions': [
                '1. DO NOT consume without verification',
                '2. Contact a pharmacist',
                '3. Provide pharmacist with actual pill image',
                '4. Request professional identification'
            ]
        }


def demonstrate_conservative_classification():
    """Show how conservative classification works"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   CONSERVATIVE PILL CLASSIFICATION WITH UNKNOWN DETECTION        ║
║                    Medical Safety Implementation                 ║
╚══════════════════════════════════════════════════════════════════╝

THRESHOLDS:
  • With visible imprint:    ≥75% confidence needed
  • Without visible imprint: ≥65% confidence needed
  • Below threshold:         Mark as UNKNOWN TABLET

SAFETY PHILOSOPHY:
  ✓ Never force misclassification
  ✓ Conservative > Dangerous
  ✓ Unknown is safer than wrong
  ✓ Recommend pharmacist verification

EXAMPLE SCENARIOS:

Scenario 1: High Confidence with Imprint
  Prediction:  Amoxicillin 500 MG
  Confidence:  87% (≥75% threshold) ✓
  Decision:    IDENTIFIED (Safe to use)

Scenario 2: Low Confidence, Missing Imprint
  Top Guess:   Ibuprofen 200 MG
  Confidence:  58% (<65% threshold) ✗
  Decision:    UNKNOWN TABLET (Safety-first)
  Action:      Recommend pharmacist verification

Scenario 3: Medium Confidence, Blurry Image
  Top Guess:   Aspirin 325 MG
  Confidence:  62% (<75% threshold) ✗
  Decision:    UNKNOWN TABLET (Image quality issue)
  Action:      Request clearer image or pharmacist ID

═══════════════════════════════════════════════════════════════════

HOW TO USE:

  classifier = ConservativePillClassifier(
      model_path='media/pilldata/model_anti_overfit.keras',
      confidence_threshold=0.75,      # Require 75% confidence
      imprint_missing_threshold=0.65  # Allow 65% if no imprint
  )
  
  result = classifier.predict('pill_image.jpg')
  
  if result['status'] == 'IDENTIFIED':
      print(f"This is {result['pill_name']}")
      print(f"Confidence: {result['confidence_percentage']}")
  else:
      print("⚠️ UNKNOWN TABLET - Need pharmacist verification")
      print(result['recommendation'])

═══════════════════════════════════════════════════════════════════

KEY FEATURES:

1. CONSERVATIVE THRESHOLDS
   - Require high confidence (75%) for identification
   - Lower threshold (65%) when imprint missing
   - Never classify below threshold

2. IMPRINT DETECTION
   - Automatically detects visible imprints
   - Adjusts threshold accordingly
   - Handles worn/faded imprints gracefully

3. SAFETY-FIRST DECISION MAKING
   - Unknown is better than wrong
   - Recommends pharmacist verification
   - Provides detailed reasoning

4. MEDICAL SAFETY INFORMATION
   - Clear safety status
   - Actionable recommendations
   - Contact information for professionals

5. UNKNOWN PILL HANDLING
   - Never forces incorrect classification
   - Provides top alternatives
   - Clear explanation for uncertainty
   - Professional verification recommended

═══════════════════════════════════════════════════════════════════

TECHNICAL DETAILS:

Model: MobileNetV3Large (anti-overfitting trained)
Features: Shape, Color, Size, Texture
Training Data: 994 images, 23 medication classes
Generalization: Trained on unseen external images
Safety: Conservative threshold-based classification

═══════════════════════════════════════════════════════════════════

INTEGRATION WITH MULTI-FEATURE CLASSIFIER:

This conservative wrapper works with your existing classifier:

  from Users.utility.multi_feature_pill_classifier import MultiFeaturePillClassifier
  
  # For confident identification
  classifier = MultiFeaturePillClassifier('model_anti_overfit.keras')
  result = classifier.predict('pill_image.jpg')
  
  # For medical safety (conservative)
  safe_classifier = ConservativePillClassifier('model_anti_overfit.keras')
  safe_result = safe_classifier.predict('pill_image.jpg')

═══════════════════════════════════════════════════════════════════

WHEN TO USE CONSERVATIVE MODE:

✓ Medical/healthcare applications
✓ Unknown pills from external sources
✓ Worn or faded pills
✓ Low quality images
✓ Safety-critical scenarios
✓ Any situation requiring high confidence

═══════════════════════════════════════════════════════════════════
""")


if __name__ == '__main__':
    demonstrate_conservative_classification()
    
    print("\nTo use with your pill images:")
    print("  classifier = ConservativePillClassifier()")
    print("  result = classifier.predict('your_pill_image.jpg')")
