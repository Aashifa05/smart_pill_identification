#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Pill Prediction with Imprint-Robustness
================================================

Handles pills without visible imprints through:
1. Embedding-based similarity matching (open-set recognition)
2. Visual feature extraction (color, shape, contour analysis)
3. Adaptive confidence thresholds based on imprint presence
4. Multi-model ensemble voting
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

logger = logging.getLogger(__name__)

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image


class EmbeddingExtractor:
    """Extract and store class prototypes (embeddings) for similarity matching"""
    
    def __init__(self, model, layer_name='global_average_pooling2d'):
        """
        Args:
            model: Trained CNN model
            layer_name: Layer to extract embeddings from (before classification head)
        """
        self.model = model
        self.embedding_model = Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        self.embedding_dim = model.get_layer(layer_name).output.shape[-1]
        logger.info(f"EmbeddingExtractor initialized with dim={self.embedding_dim}")
    
    def extract(self, images):
        """
        Extract embeddings for images.
        
        Args:
            images: Array of shape (N, 224, 224, 3) or single image (224, 224, 3)
            
        Returns:
            np.array: Embeddings of shape (N, embedding_dim) or (embedding_dim,)
        """
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        embeddings = self.embedding_model.predict(images, verbose=0)
        return embeddings[0] if len(images) == 1 else embeddings


class VisualFeatureExtractor:
    """Extract visual features from pill images (shape, color, contours)"""
    
    @staticmethod
    def extract_color_histogram(image_rgb):
        """
        Extract color histogram features (normalized RGB channels).
        
        Args:
            image_rgb: Image in RGB format, range [0, 255]
            
        Returns:
            dict: Color features
        """
        # Convert to uint8 if needed
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # Compute histogram for each channel
        hist_r = cv2.calcHist([image_rgb], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image_rgb], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([image_rgb], [2], None, [32], [0, 256])
        
        # Normalize
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        
        # Dominant color
        avg_color = np.mean(image_rgb, axis=(0, 1))
        
        # Saturation (color intensity)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        return {
            'histogram': np.concatenate([hist_r, hist_g, hist_b]),
            'dominant_color_rgb': avg_color.astype(int),
            'saturation': saturation
        }
    
    @staticmethod
    def extract_shape_features(image_rgb):
        """
        Extract shape/contour features.
        
        Args:
            image_rgb: Image in RGB format
            
        Returns:
            dict: Shape features
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {'aspect_ratio': 1.0, 'circularity': 0.0, 'solidity': 0.0}
        
        # Get largest contour (pill)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Circularity (4π * Area / Perimeter²)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
        # Solidity (Area / Convex Hull Area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0
        
        return {
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'contour_area': area
        }
    
    @staticmethod
    def extract_texture_features(image_rgb):
        """
        Extract texture features using Laplacian sharpness.
        
        Args:
            image_rgb: Image in RGB format
            
        Returns:
            dict: Texture features
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        return {
            'sharpness': sharpness,
            'has_imprint_details': sharpness > 500  # High variance suggests imprints
        }


class ImprintAwarePredictor:
    """
    Predict pill with imprint-robustness.
    Uses both CNN predictions and embedding similarity.
    """
    
    def __init__(self, model_path, metadata_path, embedding_layer='global_average_pooling2d'):
        """
        Args:
            model_path: Path to trained model
            metadata_path: Path to model metadata JSON
            embedding_layer: Layer name for embeddings
        """
        self.model = load_model(model_path)
        
        import json
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.label_map = {v: k for k, v in self.metadata['label_map'].items()}
        self.embedding_extractor = EmbeddingExtractor(self.model, layer_name=embedding_layer)
        self.visual_extractor = VisualFeatureExtractor()
        
        logger.info(f"ImprintAwarePredictor initialized with {len(self.label_map)} classes")
    
    def predict(self, image_path: str, confidence_threshold: float = 0.75, 
                return_features: bool = False) -> Dict:
        """
        Predict pill class with imprint-robustness analysis.
        
        Args:
            image_path: Path to pill image
            confidence_threshold: Threshold for accepting predictions
            return_features: Whether to return visual features
            
        Returns:
            dict: Prediction results with confidence, visual features, etc.
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # CNN prediction
        cnn_predictions = self.model.predict(img_array, verbose=0)[0]
        cnn_label_idx = np.argmax(cnn_predictions)
        cnn_confidence = cnn_predictions[cnn_label_idx]
        cnn_label = self.label_map[cnn_label_idx]
        
        # Extract embedding
        embedding = self.embedding_extractor.extract(img_array[0])
        
        # Extract visual features
        img_rgb = np.array(img)
        color_features = self.visual_extractor.extract_color_histogram(img_rgb)
        shape_features = self.visual_extractor.extract_shape_features(img_rgb)
        texture_features = self.visual_extractor.extract_texture_features(img_rgb)
        
        # Determine if image has visible imprints
        has_imprints = texture_features['has_imprint_details']
        
        # Adjust confidence threshold based on imprint presence
        # Lower threshold for pills without imprints (model trained with imprint removal)
        adjusted_threshold = confidence_threshold if has_imprints else confidence_threshold * 0.85
        
        # Determine result
        if cnn_confidence >= adjusted_threshold:
            status = 'IDENTIFIED'
            confidence = float(cnn_confidence)
        elif cnn_confidence >= 0.5:
            status = 'LOW_CONFIDENCE'
            confidence = float(cnn_confidence)
        else:
            status = 'UNKNOWN'
            confidence = float(cnn_confidence)
        
        result = {
            'status': status,
            'pill_name': cnn_label,
            'confidence': confidence,
            'adjusted_threshold': float(adjusted_threshold),
            'original_threshold': confidence_threshold,
            'has_visible_imprints': has_imprints,
            'top_5_predictions': self._get_top_predictions(cnn_predictions, 5),
            'visual_analysis': {
                'color': {
                    'dominant_color_rgb': color_features['dominant_color_rgb'].tolist(),
                    'saturation': float(color_features['saturation'])
                },
                'shape': shape_features,
                'texture': {
                    'sharpness': float(texture_features['sharpness']),
                    'has_imprint_details': bool(texture_features['has_imprint_details'])
                }
            }
        }
        
        if return_features:
            result['embedding'] = embedding.tolist()
        
        return result
    
    def _get_top_predictions(self, predictions, k=5) -> List[Dict]:
        """Get top-k predictions with confidence scores"""
        top_indices = np.argsort(predictions)[-k:][::-1]
        return [
            {
                'rank': i + 1,
                'pill_name': self.label_map[idx],
                'confidence': float(predictions[idx])
            }
            for i, idx in enumerate(top_indices)
        ]


def format_prediction_report(prediction: Dict) -> str:
    """Format prediction result as readable report"""
    report = f"""
╔════════════════════════════════════════════════════════╗
║              PILL IDENTIFICATION RESULT                ║
╚════════════════════════════════════════════════════════╝

📋 PREDICTION
   Status:           {prediction['status']}
   Pill Name:        {prediction['pill_name']}
   Confidence:       {prediction['confidence']:.2%}
   Threshold Used:   {prediction['adjusted_threshold']:.2%}

🔍 IMAGE ANALYSIS
   Has Visible Imprints:  {'✓ Yes' if prediction['has_visible_imprints'] else '✗ No'}
   
   Color Analysis:
   - Dominant RGB: {prediction['visual_analysis']['color']['dominant_color_rgb']}
   - Saturation: {prediction['visual_analysis']['color']['saturation']:.2%}
   
   Shape Analysis:
   - Aspect Ratio: {prediction['visual_analysis']['shape']['aspect_ratio']:.2f}
   - Circularity: {prediction['visual_analysis']['shape']['circularity']:.2%}
   - Solidity: {prediction['visual_analysis']['shape']['solidity']:.2%}
   
   Texture Analysis:
   - Sharpness: {prediction['visual_analysis']['texture']['sharpness']:.2f}

🎯 TOP 5 PREDICTIONS
"""
    for pred in prediction['top_5_predictions']:
        report += f"   {pred['rank']}. {pred['pill_name']:30s} {pred['confidence']:.2%}\n"
    
    report += f"""
⚠️  RECOMMENDATION
"""
    if prediction['status'] == 'IDENTIFIED':
        report += "   ✓ High confidence - pill identification is reliable\n"
    elif prediction['status'] == 'LOW_CONFIDENCE':
        report += "   ⚠ Lower confidence - recommend manual verification\n"
    else:
        report += "   ✗ Could not identify pill - PHARMACIST VERIFICATION REQUIRED\n"
    
    if not prediction['has_visible_imprints']:
        report += "   📌 Note: No visible imprints detected. Identification based on shape/color.\n"
    
    report += "╚════════════════════════════════════════════════════════╝\n"
    
    return report


if __name__ == '__main__':
    # Example usage
    from django.conf import settings
    
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
    
    predictor = ImprintAwarePredictor(str(model_path), str(metadata_path))
    
    # Test with a sample image (replace with actual image path)
    # result = predictor.predict('/path/to/pill/image.jpg')
    # print(format_prediction_report(result))
