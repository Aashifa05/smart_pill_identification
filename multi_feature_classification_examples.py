#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Practical Examples: Multi-Feature Pill Classification
=====================================================

Real-world usage examples for the multi-feature pill classifier.
Shows how to classify pills using shape, color, size, imprint, and texture.

Key Principle: Imprint is SUPPLEMENTARY, not REQUIRED for classification.
"""

import os
import sys
from pathlib import Path
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.multi_feature_pill_classifier import (
    MultiFeaturePillClassifier,
    format_comprehensive_report
)


def example_1_basic_classification():
    """Example 1: Basic pill classification"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Pill Classification")
    print("="*70)
    
    from django.conf import settings
    
    # Initialize classifier
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
    
    classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))
    
    # Classify a pill (use your own image path)
    image_path = 'path/to/pill_image.jpg'
    result = classifier.predict(image_path)
    
    # Check result
    print(f"\nPill identified as: {result['pill_name']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Status: {result['status']}")
    
    # See if imprints were used
    has_imprints = result['analysis']['has_visible_imprints']
    print(f"Imprints visible: {has_imprints}")
    
    if result['status'] == 'IDENTIFIED':
        print("✓ Successfully classified!")
    elif result['status'] == 'LOW_CONFIDENCE':
        print("⚠ Low confidence - manual review recommended")
    else:
        print("✗ Could not classify - manual verification required")


def example_2_extract_all_features():
    """Example 2: Extract and analyze all features"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Extract All Features")
    print("="*70)
    
    from django.conf import settings
    
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
    
    classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))
    result = classifier.predict('path/to/pill_image.jpg')
    
    # Extract and display all 5 feature types
    features = result['features']
    
    # 1. SHAPE Features
    print("\n1️⃣  SHAPE Features:")
    print(f"   Aspect Ratio (width/height): {features['shape']['aspect_ratio']:.2f}")
    print(f"   Circularity (0-1, 1=circle): {features['shape']['circularity']:.2f}")
    print(f"   Solidity (0-1, 1=solid):      {features['shape']['solidity']:.2f}")
    print(f"   Roundness (0-1):              {features['shape']['roundness']:.2f}")
    print(f"   Compactness:                  {features['shape']['compactness']:.2f}")
    
    # 2. COLOR Features
    print("\n2️⃣  COLOR Features:")
    rgb = features['color']['dominant_color_rgb']
    print(f"   Dominant RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})")
    print(f"   Saturation: {features['color']['saturation']:.1%}")
    print(f"   Brightness: {features['color']['brightness']:.1%}")
    print(f"   Color Uniformity: {features['color']['color_uniformity']:.1%}")
    
    # 3. SIZE Features
    print("\n3️⃣  SIZE Features:")
    print(f"   Pill Area: {features['size']['pill_area_pixels']:.0f} pixels")
    print(f"   Area Ratio: {features['size']['pill_area_ratio']:.1%}")
    print(f"   Width x Height: {features['size']['width']:.0f} x {features['size']['height']:.0f}")
    print(f"   Perimeter: {features['size']['perimeter']:.0f} pixels")
    
    # 4. TEXTURE Features
    print("\n4️⃣  TEXTURE Features:")
    print(f"   Smoothness: {features['texture']['smoothness']:.1%}")
    print(f"   Surface Uniformity: {features['texture']['surface_uniformity']:.1%}")
    print(f"   Edge Strength X: {features['texture']['edge_strength_x']:.1f}")
    print(f"   Edge Strength Y: {features['texture']['edge_strength_y']:.1f}")
    
    # 5. IMPRINT Features
    print("\n5️⃣  IMPRINT Features (supplementary):")
    print(f"   Has Visible Imprints: {features['imprint']['has_visible_imprints']}")
    print(f"   Clarity Score: {features['imprint']['clarity']:.1%}")
    print(f"   Sharpness: {features['imprint']['sharpness']:.0f}")
    print(f"   Contrast: {features['imprint']['contrast']:.1%}")


def example_3_handle_missing_imprints():
    """Example 3: Correctly handle pills without visible imprints"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Handle Pills Without Visible Imprints")
    print("="*70)
    
    from django.conf import settings
    
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
    
    classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))
    result = classifier.predict('path/to/pill_image.jpg')
    
    # DON'T do this:
    print("\n❌ WRONG WAY:")
    print("if not result['analysis']['has_visible_imprints']:")
    print("    status = 'UNKNOWN'  # This would be WRONG!")
    print("    print('Pill is unknown because imprints are missing')")
    
    # DO this:
    print("\n✅ CORRECT WAY:")
    
    has_imprints = result['analysis']['has_visible_imprints']
    confidence = result['confidence']
    adjusted_threshold = result['adjusted_threshold']
    
    print(f"Has visible imprints: {has_imprints}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Threshold (adjusted for missing imprints): {adjusted_threshold:.1%}")
    
    if confidence >= adjusted_threshold:
        status = 'IDENTIFIED'
        print(f"\n✓ Status: {status}")
        print("The classifier already accounts for missing imprints!")
        print("The threshold was automatically lowered, so high confidence")
        print("still means reliable identification.")
    elif confidence >= 0.5:
        status = 'LOW_CONFIDENCE'
        print(f"\n⚠ Status: {status}")
        print("Recommend manual review")
    else:
        status = 'UNKNOWN'
        print(f"\n✗ Status: {status}")
        print("Cannot identify - need pharmacist verification")
    
    print("\n💡 Key Point:")
    print("The classifier was trained with imprint-removal augmentation.")
    print("It learns to identify pills by shape, color, size, and texture.")
    print("Missing imprints don't automatically mean UNKNOWN!")


def example_4_batch_classify():
    """Example 4: Classify multiple pills"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Classify Multiple Pills")
    print("="*70)
    
    from django.conf import settings
    from pathlib import Path
    
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
    
    classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))
    
    # Get list of pill images
    pill_images = list(Path('path/to/pill/images').glob('*.jpg'))
    
    print(f"Classifying {len(pill_images)} pills...")
    
    results = []
    for i, image_path in enumerate(pill_images, 1):
        result = classifier.predict(str(image_path), confidence_threshold=0.75)
        results.append(result)
        
        print(f"\n{i}. {image_path.name}")
        print(f"   Pill: {result['pill_name']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Status: {result['status']}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("Summary:")
    identified = sum(1 for r in results if r['status'] == 'IDENTIFIED')
    low_conf = sum(1 for r in results if r['status'] == 'LOW_CONFIDENCE')
    unknown = sum(1 for r in results if r['status'] == 'UNKNOWN')
    
    print(f"  IDENTIFIED: {identified}/{len(results)} ({identified/len(results):.0%})")
    print(f"  LOW_CONFIDENCE: {low_conf}/{len(results)} ({low_conf/len(results):.0%})")
    print(f"  UNKNOWN: {unknown}/{len(results)} ({unknown/len(results):.0%})")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"  Average Confidence: {avg_confidence:.1%}")


def example_5_integration_with_django():
    """Example 5: Django view integration"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Django View Integration")
    print("="*70)
    
    code = '''
from django.http import JsonResponse
from django.views import View
from django.views.decorators.http import require_http_methods
from Users.utility.multi_feature_pill_classifier import (
    MultiFeaturePillClassifier,
    format_comprehensive_report
)
from pathlib import Path
from django.conf import settings
import tempfile

class PillClassificationView(View):
    """Django view for pill classification"""
    
    def __init__(self):
        super().__init__()
        model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
        metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
        self.classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))
    
    def post(self, request):
        """Handle pill image upload and classification"""
        
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            for chunk in image_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        try:
            # Classify
            result = self.classifier.predict(tmp_path, confidence_threshold=0.75)
            
            return JsonResponse({
                'status': result['status'],
                'pill_name': result['pill_name'],
                'confidence': result['confidence'],
                'has_visible_imprints': result['analysis']['has_visible_imprints'],
                'top_5': result['top_5_predictions'],
                'features_summary': {
                    'shape': result['features']['shape'],
                    'color': {k: v for k, v in result['features']['color'].items() 
                             if k != 'histogram'},
                    'size': result['features']['size'],
                    'texture': result['features']['texture'],
                    'imprint': result['features']['imprint']
                }
            })
        finally:
            import os
            os.unlink(tmp_path)

@require_http_methods(["POST"])
def classify_pill(request):
    """Standalone view function"""
    view = PillClassificationView()
    return view.post(request)
'''
    
    print(code)


def example_6_api_integration():
    """Example 6: REST API integration"""
    print("\n" + "="*70)
    print("EXAMPLE 6: REST API Integration (FastAPI/Flask)")
    print("="*70)
    
    code = '''
# Using FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from Users.utility.multi_feature_pill_classifier import MultiFeaturePillClassifier
from pathlib import Path
import tempfile

app = FastAPI()

# Initialize classifier on startup
model_path = Path('media/pilldata/model_imprint_robust.h5')
metadata_path = Path('media/pilldata/model_imprint_robust_metadata.json')
classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))

@app.post('/api/classify-pill')
async def classify_pill(file: UploadFile = File(...)):
    """Classify a pill from uploaded image"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image')
    
    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        result = classifier.predict(tmp_path, confidence_threshold=0.75)
        
        return {
            'status': result['status'],
            'pill_name': result['pill_name'],
            'confidence': float(result['confidence']),
            'has_visible_imprints': result['analysis']['has_visible_imprints'],
            'top_5_predictions': result['top_5_predictions'],
            'classification_basis': result['analysis']['classification_basis']['description']
        }
    finally:
        import os
        os.unlink(tmp_path)

@app.get('/api/pill-details/{pill_name}')
def get_pill_details(pill_name: str):
    """Get details about a pill"""
    # Implement based on your database/metadata
    pass
'''
    
    print(code)


def example_7_confidence_analysis():
    """Example 7: Analyzing confidence and thresholds"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Confidence & Threshold Analysis")
    print("="*70)
    
    from django.conf import settings
    
    model_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust.h5'
    metadata_path = Path(settings.BASE_DIR) / 'media' / 'pilldata' / 'model_imprint_robust_metadata.json'
    
    classifier = MultiFeaturePillClassifier(str(model_path), str(metadata_path))
    result = classifier.predict('path/to/pill_image.jpg')
    
    confidence = result['confidence']
    original_threshold = result['confidence_threshold']
    adjusted_threshold = result['adjusted_threshold']
    has_imprints = result['analysis']['has_visible_imprints']
    
    print(f"\nConfidence Score: {confidence:.1%}")
    print(f"Original Threshold: {original_threshold:.1%}")
    print(f"Adjusted Threshold: {adjusted_threshold:.1%}")
    print(f"Has Imprints: {has_imprints}")
    
    print(f"\n{'Interpretation:':20s}")
    
    if has_imprints:
        print(f"  ✓ Imprints are visible")
        print(f"  → Using original threshold: {original_threshold:.1%}")
        if confidence >= original_threshold:
            print(f"  ✓ Confidence {confidence:.1%} ≥ {original_threshold:.1%} = IDENTIFIED")
        else:
            print(f"  ⚠ Confidence {confidence:.1%} < {original_threshold:.1%} = Lower confidence")
    else:
        print(f"  ⚠ No visible imprints")
        threshold_reduction = original_threshold - adjusted_threshold
        print(f"  → Threshold reduced by {threshold_reduction:.1%} (15% standard reduction)")
        print(f"  → Using adjusted threshold: {adjusted_threshold:.1%}")
        if confidence >= adjusted_threshold:
            print(f"  ✓ Confidence {confidence:.1%} ≥ {adjusted_threshold:.1%} = IDENTIFIED")
            print(f"     (without this adjustment, would be below {original_threshold:.1%})")
        else:
            print(f"  ⚠ Confidence {confidence:.1%} < {adjusted_threshold:.1%} = Lower confidence")
    
    print(f"\n💡 Key Insight:")
    print(f"The threshold adjustment means the model is MORE LENIENT when")
    print(f"imprints are missing, because it knows they're optional for classification.")


def print_menu():
    """Print example menu"""
    print("\n" + "="*70)
    print("MULTI-FEATURE PILL CLASSIFIER - PRACTICAL EXAMPLES")
    print("="*70)
    print("\nChoose an example:")
    print("  1. Basic classification")
    print("  2. Extract and analyze all features")
    print("  3. Handle pills without visible imprints (important!)")
    print("  4. Batch classify multiple pills")
    print("  5. Django view integration")
    print("  6. REST API integration (FastAPI/Flask)")
    print("  7. Confidence & threshold analysis")
    print("  0. Exit")
    print()


def main():
    """Run examples"""
    while True:
        print_menu()
        choice = input("Select example (0-7): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            example_1_basic_classification()
        elif choice == '2':
            example_2_extract_all_features()
        elif choice == '3':
            example_3_handle_missing_imprints()
        elif choice == '4':
            example_4_batch_classify()
        elif choice == '5':
            example_5_integration_with_django()
        elif choice == '6':
            example_6_api_integration()
        elif choice == '7':
            example_7_confidence_analysis()
        else:
            print("Invalid choice. Try again.")
        
        if choice != '0' and choice in map(str, range(1, 8)):
            input("\nPress Enter to continue...")


if __name__ == '__main__':
    main()
