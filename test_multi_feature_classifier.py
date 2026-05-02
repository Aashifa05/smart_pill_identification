#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Multi-Feature Pill Classification System
==============================================

Validates that pills are correctly classified using shape, color, size, 
imprint (supplementary), and texture features. Ensures pills WITHOUT 
imprints are not marked as UNKNOWN.

Usage:
    python test_multi_feature_classifier.py
"""

import os
import sys
import numpy as np
from pathlib import Path
import json
import logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from Users.utility.multi_feature_pill_classifier import (
    MultiFeaturePillClassifier,
    ShapeFeatureExtractor,
    ColorFeatureExtractor,
    SizeFeatureExtractor,
    ImprintFeatureExtractor,
    TextureFeatureExtractor,
    format_comprehensive_report
)
from PIL import Image
import cv2


class MultiFeatureClassifierValidator:
    """Validate multi-feature classification system"""
    
    def __init__(self, model_path: str, metadata_path: str):
        """Initialize validator"""
        self.classifier = MultiFeaturePillClassifier(model_path, metadata_path)
        self.shape_extractor = ShapeFeatureExtractor()
        self.color_extractor = ColorFeatureExtractor()
        self.size_extractor = SizeFeatureExtractor()
        self.imprint_extractor = ImprintFeatureExtractor()
        self.texture_extractor = TextureFeatureExtractor()
        
        logger.info("✓ Validator initialized")
    
    def test_feature_extractors(self, image_path: str) -> Dict:
        """Test individual feature extractors"""
        logger.info(f"\n{'='*60}")
        logger.info("TEST: Feature Extractors")
        logger.info(f"{'='*60}")
        
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img.resize((224, 224)))
        
        results = {}
        
        # Test Shape
        try:
            shape_features = self.shape_extractor.extract(img_array)
            results['shape'] = shape_features
            logger.info("✓ Shape features extracted:")
            for key, value in shape_features.items():
                logger.info(f"  {key:20s}: {value:.4f}")
        except Exception as e:
            logger.error(f"✗ Shape extraction failed: {e}")
            results['shape'] = None
        
        # Test Color
        try:
            color_features = self.color_extractor.extract(img_array)
            results['color'] = {k: v for k, v in color_features.items() if k != 'histogram'}
            results['color']['histogram_size'] = len(color_features['histogram'])
            logger.info("✓ Color features extracted:")
            for key, value in results['color'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key:20s}: {value:.4f}" if isinstance(value, float) else f"  {key:20s}: {value}")
                elif isinstance(value, list):
                    logger.info(f"  {key:20s}: {value}")
        except Exception as e:
            logger.error(f"✗ Color extraction failed: {e}")
            results['color'] = None
        
        # Test Size
        try:
            size_features = self.size_extractor.extract(img_array)
            results['size'] = size_features
            logger.info("✓ Size features extracted:")
            for key, value in size_features.items():
                logger.info(f"  {key:20s}: {value:.2f}" if isinstance(value, float) else f"  {key:20s}: {value}")
        except Exception as e:
            logger.error(f"✗ Size extraction failed: {e}")
            results['size'] = None
        
        # Test Imprint
        try:
            imprint_features = self.imprint_extractor.extract(img_array)
            results['imprint'] = imprint_features
            logger.info("✓ Imprint features extracted:")
            for key, value in imprint_features.items():
                if isinstance(value, bool):
                    logger.info(f"  {key:20s}: {value}")
                elif isinstance(value, (int, float)):
                    logger.info(f"  {key:20s}: {value:.4f}" if isinstance(value, float) else f"  {key:20s}: {value}")
        except Exception as e:
            logger.error(f"✗ Imprint extraction failed: {e}")
            results['imprint'] = None
        
        # Test Texture
        try:
            texture_features = self.texture_extractor.extract(img_array)
            results['texture'] = texture_features
            logger.info("✓ Texture features extracted:")
            for key, value in texture_features.items():
                logger.info(f"  {key:20s}: {value:.4f}" if isinstance(value, float) else f"  {key:20s}: {value}")
        except Exception as e:
            logger.error(f"✗ Texture extraction failed: {e}")
            results['texture'] = None
        
        return results
    
    def test_full_prediction(self, image_path: str) -> Dict:
        """Test full classification pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info("TEST: Full Classification Pipeline")
        logger.info(f"{'='*60}")
        
        try:
            result = self.classifier.predict(image_path, confidence_threshold=0.75)
            
            logger.info(f"✓ Prediction successful:")
            logger.info(f"  Status:           {result['status']}")
            logger.info(f"  Pill Name:        {result['pill_name']}")
            logger.info(f"  Confidence:       {result['confidence']:.2%}")
            logger.info(f"  Threshold:        {result['adjusted_threshold']:.2%}")
            logger.info(f"  Has Imprints:     {result['analysis']['has_visible_imprints']}")
            
            # Check if imprint was required
            if not result['analysis']['has_visible_imprints'] and result['status'] == 'IDENTIFIED':
                logger.info("  ✓ PASS: Pill identified WITHOUT imprints")
            elif result['analysis']['has_visible_imprints'] and result['status'] == 'IDENTIFIED':
                logger.info("  ✓ PASS: Pill identified WITH imprints")
            
            return result
        except Exception as e:
            logger.error(f"✗ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_imprint_robustness(self, image_path: str) -> None:
        """Test that pills are NOT marked UNKNOWN just due to missing imprints"""
        logger.info(f"\n{'='*60}")
        logger.info("TEST: Imprint Robustness")
        logger.info(f"{'='*60}")
        logger.info("Verifying: Pills without imprints are NOT auto-marked as UNKNOWN")
        
        result = self.classifier.predict(image_path, confidence_threshold=0.75)
        
        if result is None:
            logger.error("✗ FAIL: Prediction failed")
            return
        
        # Check the key principle
        has_imprints = result['analysis']['has_visible_imprints']
        status = result['status']
        confidence = result['confidence']
        
        logger.info(f"\n  Image Analysis:")
        logger.info(f"  - Has visible imprints: {has_imprints}")
        logger.info(f"  - Confidence score:     {confidence:.2%}")
        logger.info(f"  - Classification:       {status}")
        
        # Test 1: If confidence is high, should be IDENTIFIED regardless of imprints
        if confidence > 0.75 and status != 'IDENTIFIED':
            logger.error(f"  ✗ FAIL: High confidence ({confidence:.1%}) but marked {status}")
        elif confidence > 0.75 and status == 'IDENTIFIED':
            logger.info(f"  ✓ PASS: High confidence = IDENTIFIED (with or without imprints)")
        
        # Test 2: Missing imprints should NOT automatically make it UNKNOWN
        if not has_imprints and confidence > 0.5 and status == 'UNKNOWN':
            logger.error(f"  ✗ FAIL: Marked UNKNOWN solely due to missing imprints (confidence: {confidence:.1%})")
        elif not has_imprints and confidence > 0.5 and status in ['IDENTIFIED', 'LOW_CONFIDENCE']:
            logger.info(f"  ✓ PASS: No imprints, but classified by other features")
        
        # Test 3: Threshold should be adjusted
        original = result['confidence_threshold']
        adjusted = result['adjusted_threshold']
        
        if not has_imprints:
            if adjusted < original:
                logger.info(f"  ✓ PASS: Threshold adjusted ({original:.1%} → {adjusted:.1%}) for missing imprints")
            else:
                logger.error(f"  ✗ FAIL: Threshold not adjusted for missing imprints")
    
    def test_feature_importance(self, image_path: str) -> None:
        """Test that all 5 features are being used"""
        logger.info(f"\n{'='*60}")
        logger.info("TEST: All 5 Features Used")
        logger.info(f"{'='*60}")
        
        result = self.classifier.predict(image_path, confidence_threshold=0.75)
        
        if result is None:
            logger.error("✗ Prediction failed")
            return
        
        features = result['features']
        all_present = True
        
        required_keys = {
            'shape': ['aspect_ratio', 'circularity', 'solidity'],
            'color': ['dominant_color_rgb', 'saturation', 'brightness'],
            'size': ['pill_area_pixels', 'width', 'height'],
            'texture': ['smoothness', 'surface_uniformity'],
            'imprint': ['has_visible_imprints', 'clarity']
        }
        
        for feature_type, required_keys_list in required_keys.items():
            if feature_type in features:
                feature_data = features[feature_type]
                missing = [k for k in required_keys_list if k not in feature_data]
                
                if missing:
                    logger.error(f"  ✗ FAIL: {feature_type.upper()} missing keys: {missing}")
                    all_present = False
                else:
                    logger.info(f"  ✓ {feature_type.upper():10s} features present")
            else:
                logger.error(f"  ✗ FAIL: {feature_type.upper()} features missing")
                all_present = False
        
        if all_present:
            logger.info("\n  ✓ PASS: All 5 primary features are present")
        else:
            logger.error("\n  ✗ FAIL: Some features are missing")
    
    def test_classification_basis(self, image_path: str) -> None:
        """Test that classification basis is documented"""
        logger.info(f"\n{'='*60}")
        logger.info("TEST: Classification Basis Documentation")
        logger.info(f"{'='*60}")
        
        result = self.classifier.predict(image_path, confidence_threshold=0.75)
        
        if result is None:
            logger.error("✗ Prediction failed")
            return
        
        basis = result['analysis'].get('classification_basis', {})
        
        if 'description' in basis:
            logger.info(f"✓ Classification basis documented:")
            logger.info(f"  \"{basis['description']}\"")
        else:
            logger.error("✗ FAIL: Classification basis not documented")
        
        if 'primary_features' in basis:
            logger.info(f"✓ Primary features used: {', '.join(basis['primary_features'])}")
        else:
            logger.error("✗ FAIL: Primary features not listed")
    
    def run_all_tests(self, image_path: str) -> None:
        """Run all validation tests"""
        logger.info("\n" + "="*60)
        logger.info("MULTI-FEATURE PILL CLASSIFIER VALIDATION")
        logger.info("="*60)
        
        # Test 1: Feature extractors
        self.test_feature_extractors(image_path)
        
        # Test 2: Full prediction
        result = self.test_full_prediction(image_path)
        
        if result:
            # Test 3: Imprint robustness
            self.test_imprint_robustness(image_path)
            
            # Test 4: Feature importance
            self.test_feature_importance(image_path)
            
            # Test 5: Classification basis
            self.test_classification_basis(image_path)
            
            # Final report
            logger.info(f"\n{'='*60}")
            logger.info("COMPREHENSIVE CLASSIFICATION REPORT")
            logger.info(f"{'='*60}")
            report = format_comprehensive_report(result)
            logger.info(report)
        
        logger.info("\n" + "="*60)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*60 + "\n")


def main():
    """Main validation entry point"""
    from django.conf import settings
    
    logger.info("Multi-Feature Pill Classifier Validation Suite")
    logger.info("=" * 60)
    
    # Check if model files exist
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
    
    if not model_path.exists():
        logger.error(f"✗ Model not found: {model_path}")
        logger.info("\nTo create the model, run: python train_imprint_robust.py")
        return
    
    if not metadata_path.exists():
        logger.error(f"✗ Metadata not found: {metadata_path}")
        return
    
    # Initialize validator
    validator = MultiFeatureClassifierValidator(str(model_path), str(metadata_path))
    
    # Find test images
    test_image_dir = Path(settings.BASE_DIR) / 'media' / 'test_images'
    
    if test_image_dir.exists():
        test_images = list(test_image_dir.glob('*.jpg')) + list(test_image_dir.glob('*.png'))
        
        if test_images:
            logger.info(f"\nFound {len(test_images)} test images")
            logger.info(f"Testing with first image: {test_images[0].name}")
            
            validator.run_all_tests(str(test_images[0]))
        else:
            logger.warning(f"No test images found in {test_image_dir}")
    else:
        logger.warning(f"Test image directory not found: {test_image_dir}")
        logger.info("\nTo test with your own image, use:")
        logger.info("  validator = MultiFeatureClassifierValidator(model_path, metadata_path)")
        logger.info("  validator.run_all_tests('/path/to/your/pill/image.jpg')")


if __name__ == '__main__':
    main()
