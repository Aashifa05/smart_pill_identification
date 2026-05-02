"""
Unlabeled Pill Detection System
Improves identification of pills without visible imprints/labels

Strategies:
1. Multi-feature analysis: color, shape, size (not just imprint)
2. Placeholder embeddings for color/shape similarity matching
3. Lower confidence threshold for pills known to have no imprint
4. Visual feature extraction for manual verification
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Pills commonly without visible imprints (store color, shape, size info)
UNLABELED_PILLS_DB = {
    'Calcitriol 0.00025 MG': {
        'typical_color': ['white', 'clear', 'translucent'],
        'typical_shape': ['oval', 'capsule'],
        'typical_size': 'very small (0.25mg is micro-dose)',
        'imprint_status': 'NO VISIBLE IMPRINT',
        'identification_confidence_modifier': 0.95,  # Slightly lower threshold for unlabeled pills
        'visual_verification_required': True,
        'color_range_hsv': {
            'white': {'h': (0, 180), 'v': (200, 255)},
            'clear': {'h': (0, 180), 'v': (180, 255), 'transparency': True}
        }
    },
    'benzonatate 100 MG': {
        'typical_color': ['yellow', 'orange'],
        'typical_shape': ['oval', 'oblong'],
        'typical_size': 'medium',
        'imprint_status': 'MINIMAL OR NO IMPRINT',
        'identification_confidence_modifier': 0.90,
        'visual_verification_required': True,
        'color_range_hsv': {
            'yellow': {'h': (15, 35), 's': (100, 255), 'v': (100, 255)},
            'orange': {'h': (5, 20), 's': (100, 255), 'v': (100, 255)}
        }
    },
    'Calcitriol 0.00025 MG': {
        'typical_color': ['white', 'colorless'],
        'typical_shape': ['round', 'oval'],
        'typical_size': 'tiny (micro-dose)',
        'imprint_status': 'USUALLY NO IMPRINT',
        'identification_confidence_modifier': 0.92,
        'visual_verification_required': True,
        'color_range_hsv': {
            'white': {'h': (0, 180), 's': (0, 50), 'v': (200, 255)},
            'colorless': {'h': (0, 180), 's': (0, 30), 'v': (180, 255)}
        }
    }
}

class UnlabeledPillDetector:
    """
    Detect and handle pills without visible imprints.
    Provides multi-feature analysis and confidence adjustment.
    """
    
    def __init__(self):
        self.unlabeled_pills = UNLABELED_PILLS_DB
        self.logger = logger
    
    def is_unlabeled_pill(self, pill_name):
        """Check if pill is known to have no/minimal imprint."""
        return pill_name in self.unlabeled_pills
    
    def get_confidence_modifier(self, pill_name):
        """
        Get confidence threshold modifier for unlabeled pills.
        
        Example: If pill typically has no imprint, allow slightly lower confidence
        because CNN won't see imprint text to identify it.
        
        Args:
            pill_name: Name of the pill
            
        Returns:
            float: Modifier to apply (0.8-1.0 range)
        """
        if pill_name in self.unlabeled_pills:
            modifier = self.unlabeled_pills[pill_name].get('identification_confidence_modifier', 0.90)
            self.logger.info(f"[UNLABELED] {pill_name}: Applying confidence modifier {modifier}")
            return modifier
        return 1.0  # No modification for labeled pills
    
    def extract_color_features(self, image_path):
        """
        Extract dominant color and color distribution from image.
        
        Args:
            image_path: Path to pill image
            
        Returns:
            dict: Color analysis results
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return {'error': 'Image load failed'}
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Find dominant color (most common HSV value)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                               [0, 180, 0, 256, 0, 256])
            
            # Get average color in BGR
            b, g, r = cv2.split(img)
            avg_color = {
                'b': int(np.mean(b)),
                'g': int(np.mean(g)),
                'r': int(np.mean(r))
            }
            
            # Classify color
            color_name = self._classify_color(avg_color)
            
            # Check for white (clear/colorless pills)
            is_white_like = (avg_color['r'] > 150 and avg_color['g'] > 150 and avg_color['b'] > 150)
            
            # Check transparency (area with very high brightness)
            transparent_ratio = np.sum(np.all(img > 220, axis=2)) / img.size
            
            return {
                'dominant_color': color_name,
                'avg_bgr': avg_color,
                'is_white_like': is_white_like,
                'transparent_ratio': transparent_ratio,
                'color_confidence': 0.75  # Color detection confidence
            }
        
        except Exception as e:
            self.logger.error(f"Color extraction error: {str(e)}")
            return {'error': str(e)}
    
    def extract_shape_features(self, image_path):
        """
        Extract shape information from pill image.
        
        Args:
            image_path: Path to pill image
            
        Returns:
            dict: Shape analysis results
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {'error': 'Image load failed'}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find contours
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'error': 'No contours found'}
            
            # Get largest contour (the pill)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate moments and properties
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # Fit ellipse
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    aspect_ratio = ellipse[1][0] / ellipse[1][1] if ellipse[1][1] != 0 else 1
                else:
                    aspect_ratio = 1.0
                
                # Classify shape
                shape_name = self._classify_shape(circularity, aspect_ratio)
                
                return {
                    'shape': shape_name,
                    'circularity': float(circularity),  # 1.0 = perfect circle
                    'aspect_ratio': float(aspect_ratio),
                    'area_pixels': int(area),
                    'shape_confidence': 0.70
                }
            else:
                return {'error': 'Could not calculate moments'}
        
        except Exception as e:
            self.logger.error(f"Shape extraction error: {str(e)}")
            return {'error': str(e)}
    
    def _classify_color(self, bgr_color):
        """Classify color from BGR values."""
        b, g, r = bgr_color['b'], bgr_color['g'], bgr_color['r']
        
        # Check if white/colorless
        if r > 150 and g > 150 and b > 150:
            return 'white/colorless'
        
        # Check if red (red channel dominant)
        if r > 150 and g < 100 and b < 100:
            return 'red'
        
        # Check if orange (red and green high, blue low)
        if r > 150 and g > 100 and g < 200 and b < 100:
            return 'orange'
        
        # Check if yellow (red and green very high, blue low)
        if r > 180 and g > 180 and b < 100:
            return 'yellow'
        
        # Check if green (green dominant)
        if g > 150 and r < 100 and b < 100:
            return 'green'
        
        # Check if blue (blue dominant)
        if b > 150 and r < 100 and g < 100:
            return 'blue'
        
        # Check if pink (red high, green moderate, blue moderate-high)
        if r > 150 and 80 < g < 140 and 100 < b < 180:
            return 'pink'
        
        # Check if brown (moderate r,g,b with r>g>b pattern)
        if 80 < r < 180 and 50 < g < 140 and 20 < b < 100 and r > g > b:
            return 'brown'
        
        # Default to gray
        return 'gray'
    
    def _classify_shape(self, circularity, aspect_ratio):
        """Classify shape based on circularity and aspect ratio."""
        if circularity > 0.85:
            return 'round'
        elif aspect_ratio > 2.0:
            return 'oblong/capsule'
        elif aspect_ratio > 1.3:
            return 'oval'
        else:
            return 'irregular'
    
    def match_pill_features(self, pill_name, color_features, shape_features):
        """
        Check if extracted features match known pill characteristics.
        
        Args:
            pill_name: Name of the pill
            color_features: Color analysis dict
            shape_features: Shape analysis dict
            
        Returns:
            dict: Match results and confidence boost/penalty
        """
        if pill_name not in self.unlabeled_pills:
            return {'match_score': 0.0, 'reasoning': 'Pill not in unlabeled database'}
        
        pill_info = self.unlabeled_pills[pill_name]
        matches = []
        penalties = []
        
        # Check color match
        if 'error' not in color_features:
            detected_color = color_features.get('dominant_color', 'unknown')
            typical_colors = pill_info.get('typical_color', [])
            
            color_match = detected_color in typical_colors or detected_color in str(typical_colors).lower()
            if color_match:
                matches.append(f"Color match: {detected_color}")
            else:
                penalties.append(f"Color mismatch: expected {typical_colors}, got {detected_color}")
        
        # Check shape match
        if 'error' not in shape_features:
            detected_shape = shape_features.get('shape', 'unknown')
            typical_shapes = pill_info.get('typical_shape', [])
            
            shape_match = detected_shape in typical_shapes or detected_shape in str(typical_shapes).lower()
            if shape_match:
                matches.append(f"Shape match: {detected_shape}")
            else:
                penalties.append(f"Shape mismatch: expected {typical_shapes}, got {detected_shape}")
        
        # Calculate match score
        total_checks = len(matches) + len(penalties)
        if total_checks > 0:
            match_score = len(matches) / total_checks
        else:
            match_score = 0.5  # Neutral if no checks possible
        
        return {
            'match_score': match_score,
            'matches': matches,
            'penalties': penalties,
            'reasoning': '; '.join(matches + penalties) if (matches + penalties) else 'Incomplete feature extraction'
        }
    
    def adjust_confidence_for_unlabeled(self, original_confidence, pill_name, 
                                       color_features=None, shape_features=None):
        """
        Adjust CNN confidence for pills known to have no imprints.
        
        Logic:
        - If pill is known unlabeled, slightly reduce threshold since CNN won't see imprint
        - If visual features match, boost confidence
        - If visual features don't match, reduce confidence
        
        Args:
            original_confidence: Original CNN prediction confidence (0-1)
            pill_name: Predicted pill name
            color_features: Extracted color features
            shape_features: Extracted shape features
            
        Returns:
            dict: Adjusted confidence and reasoning
        """
        if not self.is_unlabeled_pill(pill_name):
            return {
                'adjusted_confidence': original_confidence,
                'confidence_modifier': 1.0,
                'reasoning': f'{pill_name} is a labeled pill (has imprint)'
            }
        
        # Start with base modifier for unlabeled pills
        modifier = self.get_confidence_modifier(pill_name)
        
        # Apply visual feature matching if available
        if color_features or shape_features:
            feature_match = self.match_pill_features(pill_name, color_features or {}, shape_features or {})
            match_score = feature_match.get('match_score', 0.5)
            
            # Boost or penalize based on visual features
            if match_score > 0.75:
                modifier *= 1.05  # 5% boost for good visual match
            elif match_score < 0.25:
                modifier *= 0.85  # 15% penalty for poor visual match
            
            reasoning = feature_match.get('reasoning', '')
        else:
            reasoning = "No visual features extracted for comparison"
        
        adjusted_confidence = original_confidence * modifier
        adjusted_confidence = min(adjusted_confidence, 1.0)  # Cap at 1.0
        
        return {
            'adjusted_confidence': adjusted_confidence,
            'confidence_modifier': modifier,
            'original_confidence': original_confidence,
            'reasoning': f"Unlabeled pill detected. {reasoning}. Applied {modifier:.2f}x modifier.",
            'visual_features_available': bool(color_features or shape_features)
        }
    
    def get_verification_checklist(self, pill_name, color_features=None, shape_features=None):
        """
        Generate manual verification checklist for unlabeled pills.
        
        Args:
            pill_name: Name of predicted pill
            color_features: Color analysis dict
            shape_features: Shape analysis dict
            
        Returns:
            dict: Verification checklist for user/pharmacist
        """
        if not self.is_unlabeled_pill(pill_name):
            return {'checklist': [], 'required_verification': False}
        
        pill_info = self.unlabeled_pills[pill_name]
        checklist = []
        
        # Color verification
        typical_colors = pill_info.get('typical_color', [])
        checklist.append({
            'item': 'Color verification',
            'expected': ', '.join(typical_colors),
            'detected': color_features.get('dominant_color', 'unknown') if color_features else 'Not analyzed',
            'critical': True
        })
        
        # Shape verification
        typical_shapes = pill_info.get('typical_shape', [])
        checklist.append({
            'item': 'Shape verification',
            'expected': ', '.join(typical_shapes),
            'detected': shape_features.get('shape', 'unknown') if shape_features else 'Not analyzed',
            'critical': True
        })
        
        # Size verification
        checklist.append({
            'item': 'Size verification',
            'expected': pill_info.get('typical_size', 'Unknown'),
            'detected': 'Manual inspection required',
            'critical': True
        })
        
        # Imprint status
        checklist.append({
            'item': 'Imprint status',
            'expected': pill_info.get('imprint_status', 'Unknown'),
            'detected': 'Manual verification required',
            'critical': False
        })
        
        return {
            'checklist': checklist,
            'required_verification': pill_info.get('visual_verification_required', True),
            'verification_note': f"⚠️ {pill_name} typically has no visible imprint. Manual verification CRITICAL."
        }


def enhance_prediction_for_unlabeled_pills(prediction_result, image_path, pill_name):
    """
    Enhance prediction with unlabeled pill detection.
    
    Args:
        prediction_result: Original prediction dict
        image_path: Path to pill image
        pill_name: Predicted pill name
        
    Returns:
        dict: Enhanced prediction with visual analysis
    """
    detector = UnlabeledPillDetector()
    
    # Extract visual features
    color_features = detector.extract_color_features(image_path)
    shape_features = detector.extract_shape_features(image_path)
    
    # Adjust confidence for unlabeled pills
    original_confidence = float(prediction_result.get('confidence', '0').rstrip('%')) / 100
    confidence_adjustment = detector.adjust_confidence_for_unlabeled(
        original_confidence, pill_name, color_features, shape_features
    )
    
    # Get verification checklist
    verification_checklist = detector.get_verification_checklist(
        pill_name, color_features, shape_features
    )
    
    # Enhance prediction result
    enhanced_result = prediction_result.copy()
    enhanced_result['unlabeled_analysis'] = {
        'is_unlabeled_pill': detector.is_unlabeled_pill(pill_name),
        'color_analysis': color_features,
        'shape_analysis': shape_features,
        'confidence_adjustment': confidence_adjustment,
        'verification_checklist': verification_checklist,
        'adjusted_confidence': f"{confidence_adjustment['adjusted_confidence']*100:.2f}%"
    }
    
    # Update confidence in main result if it changed
    if confidence_adjustment['confidence_modifier'] != 1.0:
        enhanced_result['confidence'] = enhanced_result['unlabeled_analysis']['adjusted_confidence']
    
    return enhanced_result
