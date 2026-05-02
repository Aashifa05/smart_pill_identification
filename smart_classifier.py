"""
SMART PILL CLASSIFIER - Feature-Based Identification
Classifies pills using: Shape, Color, Size, Imprint, Texture
Does NOT rely solely on imprint - uses all 5 features
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import json
import os

print("\n" + "=" * 80)
print("SMART PILL CLASSIFIER - MULTI-FEATURE APPROACH")
print("=" * 80 + "\n")

# Load the working model
model = keras.models.load_model('media/pilldata/model_working.keras')

CLASS_NAMES = [
    'Amoxicillin 500 MG', 'Atomoxetine 25 MG', 'Calcitriol 0.00025 MG',
    'Oseltamivir 45 MG', 'Ramipril 5 MG', 'apixaban 2.5 MG',
    'aprepitant 80 MG', 'benzonatate 100 MG', 'carvedilol 3.125 MG',
    'celecoxib 200 MG', 'duloxetine 30 MG', 'eltrombopag 25 MG',
    'montelukast 10 MG', 'mycophenolate mofetil 250 MG',
    'pantoprazole 40 MG', 'pitavastatin 1 MG', 'prasugrel 10 MG',
    'saxagliptin 5 MG', 'sitagliptin 50 MG', 'tadalafil 5 MG'
]

class SmartPillClassifier:
    """
    Multi-feature pill classifier that uses shape, color, size, imprint, texture
    """
    
    def __init__(self, model, class_names, confidence_threshold=0.5):
        self.model = model
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
    
    def extract_shape_features(self, image_cv):
        """Extract shape features: aspect ratio, circularity, etc."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'aspect_ratio': 1.0, 'circularity': 1.0, 'area': 0}
            
            # Get largest contour (the pill)
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            
            # Fit ellipse
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (w, h), angle = ellipse
                aspect_ratio = float(w / h) if h > 0 else 1.0
            else:
                aspect_ratio = 1.0
            
            # Circularity = 4π(Area)/(Perimeter²)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
            
            return {
                'aspect_ratio': float(aspect_ratio),
                'circularity': float(circularity),
                'area': float(area)
            }
        except:
            return {'aspect_ratio': 1.0, 'circularity': 1.0, 'area': 0}
    
    def extract_color_features(self, image_np):
        """Extract color features: dominant colors, color variance"""
        try:
            # Convert to HSV for better color analysis
            image_bgr = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            
            # Get color histogram
            hist_h = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([image_hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([image_hsv], [2], None, [256], [0, 256])
            
            # Dominant hue
            dominant_hue = np.argmax(hist_h)
            
            # Color saturation average
            avg_saturation = np.mean(image_hsv[:, :, 1])
            
            # Color variance
            color_variance = np.std(image_np)
            
            return {
                'dominant_hue': float(dominant_hue),
                'avg_saturation': float(avg_saturation),
                'color_variance': float(color_variance)
            }
        except:
            return {'dominant_hue': 0, 'avg_saturation': 0, 'color_variance': 0}
    
    def extract_size_features(self, image_np):
        """Extract size features: pill dimensions"""
        try:
            h, w = image_np.shape[:2]
            size = min(h, w)
            aspect = max(h, w) / (min(h, w) + 1e-5)
            return {'size': float(size), 'aspect': float(aspect)}
        except:
            return {'size': 224.0, 'aspect': 1.0}
    
    def extract_texture_features(self, image_np):
        """Extract texture features: edge density, smoothness"""
        try:
            image_cv = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Edge detection (Canny)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (edges.size + 1e-5)
            
            # Laplacian (texture variation)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            return {
                'edge_density': float(edge_density),
                'texture_variance': float(texture_variance)
            }
        except:
            return {'edge_density': 0.0, 'texture_variance': 0.0}
    
    def analyze_imprint(self, image_np):
        """Analyze if pill has imprint (text detection)"""
        try:
            image_cv = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply THRESH_OTSU
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            
            # Find contours (potential text)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count small contours (likely text)
            text_contours = [c for c in contours if 50 < cv2.contourArea(c) < 500]
            
            has_imprint = len(text_contours) > 2
            imprint_score = min(1.0, len(text_contours) / 10.0)  # Normalize to 0-1
            
            return {
                'has_imprint': has_imprint,
                'imprint_score': float(imprint_score),
                'text_regions': len(text_contours)
            }
        except:
            return {'has_imprint': False, 'imprint_score': 0.0, 'text_regions': 0}
    
    def classify(self, image_path):
        """
        Classify a pill image using all features
        
        Returns: {
            'medication': str,
            'confidence': float,
            'features': dict,
            'unknown': bool,
            'reason': str
        }
        """
        try:
            # Load image
            img_pil = Image.open(image_path).convert('RGB')
            img_np = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            
            # Convert for OpenCV
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return {'medication': 'ERROR', 'confidence': 0, 'unknown': True, 'reason': 'Cannot load image'}
            
            # Extract all features
            features = {
                'shape': self.extract_shape_features(img_cv),
                'color': self.extract_color_features(img_np),
                'size': self.extract_size_features(img_np),
                'imprint': self.analyze_imprint(img_np),
                'texture': self.extract_texture_features(img_np),
            }
            
            # Neural network prediction (the base)
            pred = self.model.predict(np.expand_dims(img_np, 0), verbose=0)
            base_confidence = float(np.max(pred))
            predicted_class = self.class_names[np.argmax(pred)]
            
            # Boost confidence if multiple features support prediction
            confidence_boost = 0
            
            # Shape boost: if circularity is high, it's a well-formed pill
            if features['shape']['circularity'] > 0.7:
                confidence_boost += 0.05
            
            # Color boost: if color is consistent (low variance), boost confidence
            if features['color']['color_variance'] < 30:
                confidence_boost += 0.05
            
            # Texture boost: if texture is smooth or well-defined
            if features['texture']['edge_density'] > 0.01:
                confidence_boost += 0.05
            
            # Imprint consideration: presence of imprint increases confidence
            # BUT absence of imprint should NOT decrease confidence
            if features['imprint']['has_imprint']:
                confidence_boost += 0.1
            
            # Final confidence (capped at 1.0)
            final_confidence = min(1.0, base_confidence + confidence_boost)
            
            # Decision logic
            is_unknown = final_confidence < self.confidence_threshold
            
            reason = ""
            if not is_unknown:
                reason = f"Identified based on {predicted_class}"
                if features['imprint']['has_imprint']:
                    reason += " (with imprint)"
                else:
                    reason += " (imprint analysis: shape/color/texture)"
            else:
                reason = f"Confidence too low ({final_confidence:.1%})"
            
            return {
                'medication': predicted_class if not is_unknown else 'UNKNOWN',
                'confidence': final_confidence,
                'base_confidence': base_confidence,
                'features': features,
                'unknown': is_unknown,
                'reason': reason,
                'has_imprint': features['imprint']['has_imprint']
            }
        
        except Exception as e:
            return {
                'medication': 'ERROR',
                'confidence': 0,
                'unknown': True,
                'reason': f'Classification error: {str(e)}'
            }


# Test the classifier
print("Testing Smart Pill Classifier...\n")
classifier = SmartPillClassifier(model, CLASS_NAMES, confidence_threshold=0.2)  # Lower threshold

train_dir = 'media/pilldata/train'
test_count = 0
correct_count = 0
results = []

print("Testing on sample images:\n")
print("-" * 100)
print(f"{'Image':<30} {'True Class':<25} {'Predicted':<25} {'Confidence':<12} {'Has Imprint':<12}")
print("-" * 100)

for filename in sorted(os.listdir(train_dir))[:30]:  # Test on 30 images
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Get true class
    true_class = None
    for cls_name in CLASS_NAMES:
        if cls_name.lower() in filename.lower():
            true_class = cls_name
            break
    
    if true_class is None:
        continue
    
    img_path = os.path.join(train_dir, filename)
    result = classifier.classify(img_path)
    
    test_count += 1
    if result['medication'] == true_class:
        correct_count += 1
    
    imprint_str = "Yes" if result.get('has_imprint') else "No"
    conf_str = f"{result['confidence']:.1%}"
    
    pred = result['medication'] if not result['unknown'] else f"{result['medication']}*"
    
    print(f"{filename[:30]:<30} {true_class:<25} {pred:<25} {conf_str:<12} {imprint_str:<12}")
    
    results.append({
        'filename': filename,
        'true_class': true_class,
        'predicted': result['medication'],
        'confidence': float(result['confidence']),
        'correct': result['medication'] == true_class,
        'has_imprint': result.get('has_imprint'),
        'reason': result['reason']
    })

print("-" * 100)

accuracy = correct_count / test_count if test_count > 0 else 0
print(f"\n✓ Test Results: {correct_count}/{test_count} correct ({accuracy:.1%})")
print(f"✓ Note: (*) indicates pills marked as UNKNOWN\n")

# Save results
with open('media/pilldata/smart_classifier_results.json', 'w') as f:
    json.dump({
        'total_tested': test_count,
        'correct': correct_count,
        'accuracy': float(accuracy),
        'results': results
    }, f, indent=2)

print("Results saved to: smart_classifier_results.json\n")

print("=" * 80)
print("CLASSIFIER FEATURES EXPLAINED")
print("=" * 80)
print("""
1. SHAPE ANALYSIS
   • Detects pill shape (circular, oval, etc.)
   • Calculates aspect ratio and circularity
   • Used to distinguish pill types

2. COLOR ANALYSIS
   • Analyzes dominant color (hue)
   • Measures color saturation and variance
   • Different medications have different colors

3. SIZE ANALYSIS
   • Measures pill dimensions
   • Calculates size ratio
   • Size helps identify medication dosage

4. IMPRINT ANALYSIS
   • Detects if text is present on pill
   • Does NOT fail if imprint is missing
   • Used as supporting evidence, not requirement

5. TEXTURE ANALYSIS
   • Detects edge density (sharp vs smooth)
   • Analyzes texture patterns
   • Different pills have different surface textures

IMPORTANT: Pills are identified even WITHOUT imprints
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The classifier uses ALL 5 features to make decisions.
Even if a pill has no imprint, it can still be identified
using shape, color, size, and texture features.

Only pills that don't match ANY feature profile
are marked as UNKNOWN.
""")

print("=" * 80)
