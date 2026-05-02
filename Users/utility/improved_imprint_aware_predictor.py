#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Updated Prediction with Better Confidence Handling
===================================================

This version:
1. Uses the new balanced model
2. Has LOWER confidence threshold (0.50 instead of 0.75)
3. Better handling of edge cases
4. More detailed diagnostics
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import cv2
from PIL import Image


class ImprovedImprintAwarePredictor:
    """
    Improved predictor with better handling of pill classes.
    Uses LOWER confidence thresholds and better diagnostics.
    """
    
    def __init__(self, model_path, metadata_path):
        """
        Args:
            model_path: Path to trained model
            metadata_path: Path to model metadata JSON
        """
        self.model = load_model(model_path)
        
        import json
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.label_map = {v: k for k, v in self.metadata['label_map'].items()}
        logger.info(f"Predictor initialized with {len(self.label_map)} classes")
    
    def predict(self, image_path: str, confidence_threshold: float = 0.50, 
                return_diagnostics: bool = False) -> Dict:
        """
        Predict pill class with improved handling.
        
        Args:
            image_path: Path to pill image
            confidence_threshold: Threshold for accepting predictions (LOWER than v1!)
            return_diagnostics: Return detailed diagnostics
            
        Returns:
            dict: Prediction results
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)[0]
        top_idx = np.argmax(predictions)
        top_confidence = predictions[top_idx]
        top_label = self.label_map[top_idx]
        
        # Get top 5
        top_5_idx = np.argsort(predictions)[-5:][::-1]
        top_5 = [
            {
                'rank': i + 1,
                'pill_name': self.label_map[idx],
                'confidence': float(predictions[idx])
            }
            for i, idx in enumerate(top_5_idx)
        ]
        
        # Decision logic (LOWER threshold = better for learning)
        if top_confidence >= confidence_threshold:
            status = 'IDENTIFIED'
        elif top_confidence >= 0.30:  # Still show if reasonably confident
            status = 'LOW_CONFIDENCE'
        else:
            status = 'UNKNOWN'
        
        result = {
            'status': status,
            'pill_name': top_label,
            'confidence': float(top_confidence),
            'threshold_used': confidence_threshold,
            'top_5_predictions': top_5,
        }
        
        if return_diagnostics:
            result['diagnostics'] = {
                'all_predictions': {self.label_map[i]: float(predictions[i]) 
                                    for i in range(len(predictions))},
                'top_5_probabilities': top_5
            }
        
        return result


def compare_models(image_path, old_model_path, old_metadata_path, 
                   new_model_path, new_metadata_path):
    """
    Compare old v1 model vs new v2 model on same image.
    Shows if v2 improved.
    """
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON: v1 vs v2")
    logger.info("="*60)
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Old model (v1)
    try:
        old_model = load_model(old_model_path)
        old_pred = old_model.predict(img_array, verbose=0)[0]
        old_top_idx = np.argmax(old_pred)
        old_confidence = old_pred[old_top_idx]
        logger.info(f"v1 Model: Confidence={old_confidence:.1%}")
    except:
        logger.warning("Could not load v1 model for comparison")
        old_confidence = None
    
    # New model (v2)
    try:
        new_model = load_model(new_model_path)
        new_pred = new_model.predict(img_array, verbose=0)[0]
        new_top_idx = np.argmax(new_pred)
        new_confidence = new_pred[new_top_idx]
        logger.info(f"v2 Model: Confidence={new_confidence:.1%}")
    except:
        logger.warning("Could not load v2 model for comparison")
        new_confidence = None
    
    # Compare
    if old_confidence is not None and new_confidence is not None:
        improvement = new_confidence - old_confidence
        direction = "↑" if improvement > 0 else "↓"
        logger.info(f"Change: {direction} {abs(improvement):.1%}")


if __name__ == '__main__':
    from django.conf import settings
    
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_v2.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_v2_metadata.json'
    
    if model_path.exists():
        predictor = ImprovedImprintAwarePredictor(str(model_path), str(metadata_path))
        logger.info("Improved predictor ready!")
        
        # Test with a sample image
        # result = predictor.predict('/path/to/image.jpg', confidence_threshold=0.50)
        # logger.info(f"Result: {result}")
    else:
        logger.warning(f"Model not found at {model_path}")
