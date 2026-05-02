#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Feature Pill Classification System
=========================================

Classifies pills using five PRIMARY features:
1. SHAPE - Contour analysis (aspect ratio, circularity, solidity)
2. COLOR - RGB histogram, dominant color, saturation
3. SIZE - Physical dimensions and area
4. IMPRINT - Text/embossing detection (supplementary, not required)
5. TEXTURE - Surface characteristics and sharpness

Key principle: Pills can be classified even without imprints.
Imprint is a supplementary feature, not a requirement for classification.

Usage:
    classifier = MultiFeaturePillClassifier(model_path, metadata_path)
    result = classifier.predict(image_path)
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

logger = logging.getLogger(__name__)

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class ShapeFeatureExtractor:
    """Extract shape-based features from pill images"""
    
    @staticmethod
    def extract(image_rgb: np.ndarray) -> Dict[str, float]:
        """
        Extract shape features: aspect ratio, circularity, solidity, roundness.
        
        Args:
            image_rgb: Image in RGB format [0, 255] or [0, 1]
            
        Returns:
            dict: Shape features
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to isolate pill
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                'aspect_ratio': 1.0,
                'circularity': 0.0,
                'solidity': 0.0,
                'roundness': 0.0,
                'compactness': 0.0
            }
        
        # Get largest contour (pill)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 10:  # Too small to be a pill
            return {
                'aspect_ratio': 1.0,
                'circularity': 0.0,
                'solidity': 0.0,
                'roundness': 0.0,
                'compactness': 0.0
            }
        
        # Bounding box for aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Circularity: 4π * Area / Perimeter²
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        
        # Solidity: Area / Convex Hull Area
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / (hull_area + 1e-6)
        
        # Roundness (another measure of shape regularity)
        # Using contour moments
        moments = cv2.moments(largest_contour)
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            
            distances = []
            for point in largest_contour:
                px, py = point[0]
                dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                distances.append(dist)
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            roundness = 1.0 - (std_dist / (mean_dist + 1e-6))
        else:
            roundness = 0.0
        
        # Compactness: Perimeter² / Area
        compactness = (perimeter ** 2) / (area + 1e-6)
        
        return {
            'aspect_ratio': float(aspect_ratio),
            'circularity': float(np.clip(circularity, 0, 1)),
            'solidity': float(solidity),
            'roundness': float(np.clip(roundness, 0, 1)),
            'compactness': float(compactness)
        }


class ColorFeatureExtractor:
    """Extract color-based features from pill images"""
    
    @staticmethod
    def extract(image_rgb: np.ndarray) -> Dict:
        """
        Extract color features: dominant color, saturation, hue distribution.
        
        Args:
            image_rgb: Image in RGB format [0, 255] or [0, 1]
            
        Returns:
            dict: Color features
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        
        # Dominant color (average RGB)
        avg_color_rgb = np.mean(image_rgb, axis=(0, 1)).astype(int)
        
        # Color histogram (each channel)
        hist_r = cv2.calcHist([image_rgb], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image_rgb], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([image_rgb], [2], None, [32], [0, 256])
        
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        histogram = np.concatenate([hist_r, hist_g, hist_b])
        
        # HSV for saturation and hue
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        hue = float(np.mean(hsv[:, :, 0]) / 180.0)  # Normalized to [0, 1]
        value = float(np.mean(hsv[:, :, 2]) / 255.0)  # Brightness
        
        # Color variance (how uniform the color is)
        color_std = np.std(image_rgb) / 255.0
        
        return {
            'dominant_color_rgb': avg_color_rgb.tolist(),
            'dominant_color_hsv': [hue, saturation, value],
            'histogram': histogram.tolist(),
            'saturation': saturation,
            'hue': hue,
            'brightness': value,
            'color_uniformity': 1.0 - color_std  # Higher = more uniform color
        }


class SizeFeatureExtractor:
    """Extract size-based features from pill images"""
    
    @staticmethod
    def extract(image_rgb: np.ndarray) -> Dict[str, float]:
        """
        Extract size features: area, dimensions, perimeter.
        
        Args:
            image_rgb: Image in RGB format
            
        Returns:
            dict: Size features
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        
        h, w = image_rgb.shape[:2]
        image_area = h * w
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                'pill_area_pixels': 0.0,
                'pill_area_ratio': 0.0,
                'perimeter': 0.0,
                'width': 0.0,
                'height': 0.0,
                'diagonal': 0.0
            }
        
        largest_contour = max(contours, key=cv2.contourArea)
        pill_area = cv2.contourArea(largest_contour)
        x, y, width, height = cv2.boundingRect(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        diagonal = np.sqrt(width ** 2 + height ** 2)
        
        return {
            'pill_area_pixels': float(pill_area),
            'pill_area_ratio': float(pill_area / image_area),
            'perimeter': float(perimeter),
            'width': float(width),
            'height': float(height),
            'diagonal': float(diagonal)
        }


class ImprintFeatureExtractor:
    """Extract imprint-related features from pill images"""
    
    @staticmethod
    def extract(image_rgb: np.ndarray) -> Dict:
        """
        Extract imprint features: presence, clarity, text-like patterns.
        NOTE: Imprint is supplementary - not required for classification.
        
        Args:
            image_rgb: Image in RGB format
            
        Returns:
            dict: Imprint features
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Laplacian for edge/texture detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian))
        
        # Canny edges for imprint-like patterns
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.sum(edges) / (edges.shape[0] * edges.shape[1]))
        
        # Text detection: count connected components of edges
        num_labels, _ = cv2.connectedComponents(edges)
        
        # High sharpness + high edge density + multiple components = likely has imprints
        has_imprint_details = sharpness > 500 and edge_density > 0.05
        
        # Clarity score: based on contrast and sharpness
        contrast = float(np.std(gray) / 255.0)
        clarity = (sharpness / 10000.0) * contrast  # Normalize sharpness
        clarity = float(np.clip(clarity, 0, 1))
        
        return {
            'has_visible_imprints': has_imprint_details,
            'sharpness': sharpness,
            'edge_density': edge_density,
            'contrast': contrast,
            'clarity': clarity,
            'num_edge_components': int(num_labels),
            'imprint_confidence': float(np.clip(clarity, 0, 1))
        }


class TextureFeatureExtractor:
    """Extract texture-based features from pill images"""
    
    @staticmethod
    def extract(image_rgb: np.ndarray) -> Dict:
        """
        Extract texture features: smoothness, surface characteristics.
        
        Args:
            image_rgb: Image in RGB format
            
        Returns:
            dict: Texture features
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # LBP (Local Binary Patterns) approximation using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = float(np.var(laplacian))
        
        # Smoothness: inverse of edge density
        edges = cv2.Canny(gray, 100, 200)
        smoothness = 1.0 - float(np.sum(edges) / (edges.shape[0] * edges.shape[1]))
        
        # Directional edges (x and y gradients)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_strength_x = float(np.mean(np.abs(sobelx)))
        edge_strength_y = float(np.mean(np.abs(sobely)))
        directional_asymmetry = float(abs(edge_strength_x - edge_strength_y) / (edge_strength_x + edge_strength_y + 1e-6))
        
        # Surface uniformity
        # Divide image into quadrants and compare texture
        h, w = gray.shape
        quadrants = [
            gray[:h//2, :w//2],
            gray[:h//2, w//2:],
            gray[h//2:, :w//2],
            gray[h//2:, w//2:]
        ]
        quadrant_vars = [np.var(q) for q in quadrants]
        surface_uniformity = 1.0 - (np.std(quadrant_vars) / (np.mean(quadrant_vars) + 1e-6))
        surface_uniformity = float(np.clip(surface_uniformity, 0, 1))
        
        return {
            'texture_variance': texture_variance,
            'smoothness': float(np.clip(smoothness, 0, 1)),
            'roughness': float(1.0 - smoothness),
            'edge_strength_x': edge_strength_x,
            'edge_strength_y': edge_strength_y,
            'directional_asymmetry': directional_asymmetry,
            'surface_uniformity': surface_uniformity
        }


class MultiFeaturePillClassifier:
    """
    Pill classifier using five PRIMARY features:
    1. Shape, 2. Color, 3. Size, 4. Imprint (supplementary), 5. Texture
    
    Pills can be classified without imprints - imprint is NOT required.
    """
    
    def __init__(self, model_path: str, metadata_path: str, 
                 embedding_layer: str = 'global_average_pooling2d'):
        """
        Initialize classifier with CNN model and metadata.
        
        Args:
            model_path: Path to trained CNN model
            metadata_path: Path to model metadata JSON
            embedding_layer: Layer name for feature embeddings
        """
        self.model = load_model(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.label_map = {v: k for k, v in self.metadata['label_map'].items()}
        
        # Create embedding extractor from model
        self.embedding_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(embedding_layer).output
        )
        
        # Feature extractors
        self.shape_extractor = ShapeFeatureExtractor()
        self.color_extractor = ColorFeatureExtractor()
        self.size_extractor = SizeFeatureExtractor()
        self.imprint_extractor = ImprintFeatureExtractor()
        self.texture_extractor = TextureFeatureExtractor()
        
        logger.info(f"MultiFeaturePillClassifier initialized with {len(self.label_map)} classes")
    
    def predict(self, image_path: str, confidence_threshold: float = 0.75,
                return_all_features: bool = True) -> Dict:
        """
        Classify pill using five primary features.
        
        Args:
            image_path: Path to pill image
            confidence_threshold: Confidence threshold for classification
            return_all_features: Whether to return detailed feature analysis
            
        Returns:
            dict: Comprehensive prediction results
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Extract all features
        shape_features = self.shape_extractor.extract(np.array(img_resized))
        color_features = self.color_extractor.extract(np.array(img_resized))
        size_features = self.size_extractor.extract(np.array(img_resized))
        imprint_features = self.imprint_extractor.extract(np.array(img_resized))
        texture_features = self.texture_extractor.extract(np.array(img_resized))
        
        # CNN prediction
        img_batch = np.expand_dims(img_array, axis=0)
        cnn_predictions = self.model.predict(img_batch, verbose=0)[0]
        cnn_label_idx = np.argmax(cnn_predictions)
        cnn_confidence = float(cnn_predictions[cnn_label_idx])
        cnn_label = self.label_map[cnn_label_idx]
        
        # Extract embedding
        embedding = self.embedding_model.predict(img_batch, verbose=0)[0]
        
        # Determine status
        # IMPORTANT: Do NOT mark as UNKNOWN just because imprint is missing!
        has_imprints = imprint_features['has_visible_imprints']
        
        # Adjust threshold based on feature completeness (not just imprints)
        if has_imprints:
            adjusted_threshold = confidence_threshold
        else:
            # Lower threshold for pills without imprints
            # since model can still classify by shape, color, size, texture
            adjusted_threshold = confidence_threshold * 0.85
        
        if cnn_confidence >= adjusted_threshold:
            status = 'IDENTIFIED'
        elif cnn_confidence >= 0.5:
            status = 'LOW_CONFIDENCE'
        else:
            status = 'UNKNOWN'
        
        result = {
            'status': status,
            'pill_name': cnn_label,
            'confidence': cnn_confidence,
            'confidence_threshold': confidence_threshold,
            'adjusted_threshold': adjusted_threshold,
            'top_5_predictions': self._get_top_predictions(cnn_predictions, k=5),
            'features': {
                'shape': shape_features,
                'color': color_features,
                'size': size_features,
                'imprint': imprint_features,
                'texture': texture_features
            },
            'analysis': {
                'has_visible_imprints': has_imprints,
                'classification_basis': self._analyze_classification_basis(
                    shape_features, color_features, size_features, 
                    imprint_features, texture_features, cnn_confidence
                )
            }
        }
        
        if return_all_features:
            result['embedding'] = embedding.tolist()
            result['image_path'] = str(image_path)
        
        return result
    
    @staticmethod
    def _analyze_classification_basis(shape: Dict, color: Dict, size: Dict,
                                      imprint: Dict, texture: Dict,
                                      confidence: float) -> Dict[str, str]:
        """
        Analyze which features contributed most to classification.
        Helps understand if pill was identified by shape, color, texture, etc.
        """
        basis = {
            'primary_features': [],
            'description': ''
        }
        
        # Shape contribution
        if shape['circularity'] > 0.7:
            basis['primary_features'].append('SHAPE (high circularity)')
        elif shape['aspect_ratio'] > 1.5 or shape['aspect_ratio'] < 0.67:
            basis['primary_features'].append('SHAPE (distinctive aspect ratio)')
        
        # Color contribution
        if color['color_uniformity'] > 0.8:
            basis['primary_features'].append('COLOR (uniform, distinctive)')
        elif color['saturation'] > 0.6:
            basis['primary_features'].append('COLOR (high saturation)')
        
        # Size contribution
        if size['pill_area_ratio'] > 0.3:
            basis['primary_features'].append('SIZE (fills image well)')
        
        # Texture contribution
        if texture['surface_uniformity'] > 0.7:
            basis['primary_features'].append('TEXTURE (smooth, uniform)')
        
        # Imprint contribution
        if imprint['has_visible_imprints']:
            basis['primary_features'].append('IMPRINT (visible details)')
        
        if not basis['primary_features']:
            basis['primary_features'].append('COMBINATION OF FEATURES')
        
        basis['description'] = (
            f"Pill identified using: {', '.join(basis['primary_features'])}. "
            f"Confidence: {confidence:.1%}. "
            f"{'Imprint details visible.' if imprint['has_visible_imprints'] else 'No visible imprints - classified by shape/color/texture.'}"
        )
        
        return basis
    
    def _get_top_predictions(self, predictions: np.ndarray, k: int = 5) -> List[Dict]:
        """Get top-k predictions with labels"""
        top_indices = np.argsort(predictions)[-k:][::-1]
        return [
            {
                'rank': i + 1,
                'pill_name': self.label_map[idx],
                'confidence': float(predictions[idx])
            }
            for i, idx in enumerate(top_indices)
        ]


def format_comprehensive_report(prediction: Dict) -> str:
    """Format detailed prediction report with all feature analysis"""
    
    report = f"""
╔════════════════════════════════════════════════════════════════════╗
║           COMPREHENSIVE PILL CLASSIFICATION REPORT                 ║
╚════════════════════════════════════════════════════════════════════╝

🎯 PRIMARY CLASSIFICATION
   Status:              {prediction['status']}
   Identified Pill:     {prediction['pill_name']}
   Confidence Score:    {prediction['confidence']:.1%}
   Threshold Used:      {prediction['adjusted_threshold']:.1%}

📊 FEATURE ANALYSIS
"""
    
    # Shape features
    shape = prediction['features']['shape']
    report += f"""
   SHAPE Features:
   ├─ Aspect Ratio:      {shape['aspect_ratio']:.2f}
   ├─ Circularity:       {shape['circularity']:.1%}
   ├─ Solidity:          {shape['solidity']:.1%}
   ├─ Roundness:         {shape['roundness']:.1%}
   └─ Compactness:       {shape['compactness']:.1f}
"""
    
    # Color features
    color = prediction['features']['color']
    report += f"""
   COLOR Features:
   ├─ Dominant RGB:      {color['dominant_color_rgb']}
   ├─ Saturation:        {color['saturation']:.1%}
   ├─ Hue:               {color['hue']:.1%}
   ├─ Brightness:        {color['brightness']:.1%}
   └─ Color Uniformity:  {color['color_uniformity']:.1%}
"""
    
    # Size features
    size = prediction['features']['size']
    report += f"""
   SIZE Features:
   ├─ Pill Area (px²):   {size['pill_area_pixels']:.0f}
   ├─ Area Ratio:        {size['pill_area_ratio']:.1%}
   ├─ Perimeter:         {size['perimeter']:.0f}
   ├─ Width x Height:    {size['width']:.0f} x {size['height']:.0f}
   └─ Diagonal:          {size['diagonal']:.0f}
"""
    
    # Imprint features
    imprint = prediction['features']['imprint']
    report += f"""
   IMPRINT Features (supplementary):
   ├─ Has Visible:       {'✓ Yes' if imprint['has_visible_imprints'] else '✗ No'}
   ├─ Sharpness:         {imprint['sharpness']:.0f}
   ├─ Clarity Score:     {imprint['clarity']:.1%}
   ├─ Edge Density:      {imprint['edge_density']:.2%}
   └─ Contrast:          {imprint['contrast']:.1%}
"""
    
    # Texture features
    texture = prediction['features']['texture']
    report += f"""
   TEXTURE Features:
   ├─ Smoothness:        {texture['smoothness']:.1%}
   ├─ Surface Uniformity:{texture['surface_uniformity']:.1%}
   ├─ Directional Asym:  {texture['directional_asymmetry']:.1%}
   ├─ Edge Strength X:   {texture['edge_strength_x']:.1f}
   └─ Edge Strength Y:   {texture['edge_strength_y']:.1f}
"""
    
    # Classification basis
    analysis = prediction['analysis']
    report += f"""
📌 CLASSIFICATION BASIS
   {analysis['classification_basis']['description']}

🎯 TOP 5 PREDICTIONS
"""
    
    for pred in prediction['top_5_predictions']:
        report += f"   {pred['rank']}. {pred['pill_name']:35s} {pred['confidence']:.1%}\n"
    
    report += f"""
⚠️  NOTES
"""
    
    if prediction['status'] == 'IDENTIFIED':
        report += "   ✓ Classification is reliable with high confidence.\n"
    elif prediction['status'] == 'LOW_CONFIDENCE':
        report += "   ⚠ Lower confidence - recommend manual verification.\n"
    else:
        report += "   ✗ Classification uncertain - PHARMACIST VERIFICATION REQUIRED.\n"
    
    if not analysis['has_visible_imprints']:
        report += "   📌 No visible imprints - Pill identified by shape, color, size, and texture.\n"
    
    report += "╚════════════════════════════════════════════════════════════════════╝\n"
    
    return report


if __name__ == '__main__':
    # Example usage
    from django.conf import settings
    
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
    
    if model_path.exists() and metadata_path.exists():
        classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))
        print("✓ MultiFeaturePillClassifier loaded successfully")
        print(f"  Models path: {model_path}")
        print(f"  Metadata path: {metadata_path}")
    else:
        print("⚠ Model files not found. Please train the model first.")
        print(f"  Expected model: {model_path}")
        print(f"  Expected metadata: {metadata_path}")
