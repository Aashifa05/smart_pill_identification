#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Integration Guide - Medical Safe Pill Classifier
========================================================

This script shows how to integrate the improved classifier into your Django app.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

import logging
logger = logging.getLogger(__name__)

from medical_safe_pill_classifier import (
    MedicalSafeEnsembleClassifier,
    PillClassificationReport
)


class PillIdentificationService:
    """Service for integrating classifier into Django views"""
    
    def __init__(self):
        """Initialize the classifier service"""
        
        # Load models - use all available models for ensemble
        model_paths = [
            'media/pilldata/model_feature_learning.h5',
            'media/pilldata/model.keras',
            'media/pilldata/model_enhanced.keras',
        ]
        
        # Filter existing models
        self.available_models = [p for p in model_paths if os.path.exists(p)]
        
        if not self.available_models:
            logger.warning("No trained models found. Please train first.")
            self.classifier = None
            return
        
        # Initialize classifier
        try:
            self.classifier = MedicalSafeEnsembleClassifier(
                model_paths=self.available_models,
                metadata_path='media/pilldata/model_feature_learning_metadata.json',
                confidence_threshold=0.80  # HIGH threshold for medical safety
            )
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            self.classifier = None
    
    def identify_pill(self, image_path: str) -> dict:
        """
        Identify a pill from an image.
        
        Args:
            image_path: Path to pill image
            
        Returns:
            dict: Prediction result with safety information
        """
        if not self.classifier:
            return {
                'status': 'ERROR',
                'message': 'Classifier not initialized. Please train a model first.',
                'tablet_name': 'Error',
                'confidence': 0.0
            }
        
        try:
            prediction = self.classifier.predict(image_path)
            
            return {
                'status': prediction.status,  # 'IDENTIFIED', 'UNCERTAIN', 'UNKNOWN'
                'tablet_name': prediction.tablet_name,
                'confidence': prediction.confidence,
                'reason': prediction.reason,
                'risk_level': prediction.risk_level,  # 'SAFE', 'CAUTION', 'REJECT'
                'top_5': prediction.top_5,
                'features': prediction.features
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'status': 'ERROR',
                'message': str(e),
                'tablet_name': 'Error',
                'confidence': 0.0
            }
    
    def get_detailed_report(self, image_path: str) -> str:
        """
        Get a detailed medical report for a pill prediction.
        
        Args:
            image_path: Path to pill image
            
        Returns:
            str: Detailed text report
        """
        if not self.classifier:
            return "Classifier not initialized."
        
        prediction = self.classifier.predict(image_path)
        report = PillClassificationReport.generate_prediction_report(prediction, image_path)
        
        return report


# ============================================================================
# DJANGO VIEW INTEGRATION EXAMPLE
# ============================================================================

# In your Django views.py:
"""
from django.shortcuts import render
from django.http import JsonResponse
from PIL_integration import PillIdentificationService

# Initialize service (do this once at startup)
pill_service = PillIdentificationService()

def identify_pill_view(request):
    if request.method == 'POST':
        # Get uploaded image
        uploaded_file = request.FILES.get('image')
        
        if not uploaded_file:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        # Save temporarily
        temp_path = f'/tmp/{uploaded_file.name}'
        with open(temp_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        # Identify pill
        result = pill_service.identify_pill(temp_path)
        
        # Add medical safety guidance
        if result['risk_level'] == 'SAFE':
            result['guidance'] = 'Pill identification is RELIABLE. Safe to use.'
        elif result['risk_level'] == 'CAUTION':
            result['guidance'] = 'CAUTION: Human review required before using this identification.'
        else:  # REJECT
            result['guidance'] = 'REJECTED: Cannot identify pill. Request additional images.'
        
        return JsonResponse(result)
    
    return render(request, 'pill_identify.html')

def pill_report_view(request):
    image_path = request.GET.get('image_path')
    
    if not image_path:
        return JsonResponse({'error': 'No image path provided'}, status=400)
    
    report = pill_service.get_detailed_report(image_path)
    
    return JsonResponse({'report': report})
"""


# ============================================================================
# TESTING THE INTEGRATION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logger.info("=" * 80)
    logger.info("PILL IDENTIFICATION SERVICE - INTEGRATION TEST")
    logger.info("=" * 80)
    
    # Initialize service
    service = PillIdentificationService()
    
    # Check if classifier is ready
    if not service.classifier:
        logger.error("Classifier not ready. Please train a model first.")
        logger.info("\nTo train a model, run:")
        logger.info("  python train_feature_learning.py")
        sys.exit(1)
    
    logger.info(f"✓ Service initialized with {len(service.available_models)} models")
    logger.info(f"✓ Available models: {service.available_models}")
    
    # Test prediction if image provided
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        
        if not os.path.exists(test_image):
            logger.error(f"Image not found: {test_image}")
            sys.exit(1)
        
        logger.info(f"\nPredicting on: {test_image}")
        logger.info("-" * 80)
        
        result = service.identify_pill(test_image)
        
        logger.info(f"Status: {result['status']}")
        logger.info(f"Tablet: {result['tablet_name']}")
        logger.info(f"Confidence: {result['confidence']:.2%}")
        logger.info(f"Risk Level: {result['risk_level']}")
        logger.info(f"Reason: {result['reason']}")
        
        if result['top_5']:
            logger.info("\nTop 5 Candidates:")
            for item in result['top_5']:
                logger.info(f"  {item['rank']}. {item['tablet_name']:30} {item['confidence']:.2%}")
        
        # Detailed report
        logger.info("\n" + "=" * 80)
        report = service.get_detailed_report(test_image)
        print(report)
    else:
        logger.info("""
USAGE:
======

1. Basic initialization:
   service = PillIdentificationService()

2. Identify a pill:
   result = service.identify_pill('path/to/pill.jpg')
   print(result)

3. Get detailed report:
   report = service.get_detailed_report('path/to/pill.jpg')
   print(report)

4. Test with image:
   python pill_integration.py path/to/pill.jpg

EXPECTED OUTPUT:
================
{
    'status': 'IDENTIFIED' | 'UNCERTAIN' | 'UNKNOWN',
    'tablet_name': 'Aspirin',
    'confidence': 0.95,
    'risk_level': 'SAFE' | 'CAUTION' | 'REJECT',
    'reason': 'High confidence match',
    'top_5': [...],
    'features': {...}
}

MEDICAL SAFETY RULES:
=====================
• Status IDENTIFIED + Risk SAFE       → Use prediction
• Status UNCERTAIN + Risk CAUTION     → Require human review
• Status UNKNOWN + Risk REJECT        → Cannot identify

""")
