"""
Complete Pill Information Extraction System
Extracts: Name, Dosage, Usage, Consumption, Precaution
From: Pill Image + OCR + Database Lookup
"""

import os
import sqlite3
import json
import re
from datetime import datetime

# Check for pytesseract (optional - will fallback if not available)
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️ pytesseract not installed. Install with: pip install pytesseract")

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class PillInfoExtractor:
    """Complete pill information extraction system"""
    
    def __init__(self, model_path, database_path=None):
        """
        Initialize extractor
        
        Args:
            model_path: Path to trained MobileNetV3 model
            database_path: Path to pills database (optional)
        """
        self.model_path = model_path
        self.database_path = database_path
        
        # Load model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ Loaded model: {model_path}")
        else:
            self.model = None
            print(f"⚠️ Model not found: {model_path}")
        
        # Create database if needed
        if database_path is None:
            self.database_path = 'pills_database.db'
        
        self._ensure_database_exists()
        
        # Pill class names (22 pills from training)
        self.pill_names = [
            'Amoxicillin',
            'Ibuprofen',
            'Aspirin',
            'Acetaminophen',
            'Metformin',
            'Lisinopril',
            'Atorvastatin',
            'Omeprazole',
            'Losartan',
            'Simvastatin',
            'Metoprolol',
            'Sertraline',
            'Albuterol',
            'Levothyroxine',
            'Warfarin',
            'Insulin',
            'Cetirizine',
            'Ranitidine',
            'Ciprofloxacin',
            'Fluconazole',
            'Tramadol',
            'Vitamin-D'
        ]
    
    def _ensure_database_exists(self):
        """Create database with sample data if not exists"""
        if os.path.exists(self.database_path):
            return
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    dosage TEXT NOT NULL,
                    usage TEXT,
                    consumption TEXT,
                    precaution TEXT,
                    side_effects TEXT,
                    manufacturer TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add sample pills
            sample_pills = [
                ('Amoxicillin', '500 MG', 'Bacterial infection treatment', 
                 'Take 1 tablet every 8 hours with food', 
                 'Not safe during pregnancy. Avoid if allergic to penicillin',
                 'Nausea, diarrhea, rash', 'GlaxoSmithKline'),
                
                ('Ibuprofen', '200 MG', 'Pain relief and anti-inflammatory',
                 'Take 1-2 tablets every 4-6 hours with food',
                 'Not suitable for people with ulcers or heart issues',
                 'Stomach upset, heartburn, dizziness', 'Various'),
                
                ('Aspirin', '325 MG', 'Pain relief and blood thinner',
                 'Take 1 tablet every 4-6 hours',
                 'Not for people with bleeding disorders. Avoid during pregnancy',
                 'Stomach irritation, bleeding risk', 'Bayer'),
                
                ('Acetaminophen', '500 MG', 'Pain and fever reduction',
                 'Take 1-2 tablets every 4-6 hours. Max 4000 MG daily',
                 'Not safe if liver disease. Avoid alcohol',
                 'Liver damage if overdosed, nausea', 'Tylenol'),
                
                ('Metformin', '500 MG', 'Type 2 diabetes management',
                 'Take 1 tablet with meals, twice daily',
                 'Not for kidney failure. Regular monitoring needed',
                 'Diarrhea, nausea, metallic taste', 'Generic'),
            ]
            
            for pill in sample_pills:
                cursor.execute('''
                    INSERT INTO pills (name, dosage, usage, consumption, precaution, side_effects, manufacturer)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', pill)
            
            conn.commit()
            conn.close()
            print(f"✓ Created database: {self.database_path}")
            
        except Exception as e:
            print(f"⚠️ Database error: {e}")
    
    # ========================================================================
    # STEP 1: VISUAL CLASSIFICATION
    # ========================================================================
    
    def classify_by_image(self, image_path):
        """
        Step 1: Classify pill using MobileNetV3
        
        Args:
            image_path: Path to pill image
            
        Returns:
            dict: {pill_name, confidence, method}
        """
        if self.model is None:
            return {
                'pill_name': 'Unknown',
                'confidence': 0.0,
                'method': 'visual_classification',
                'error': 'Model not loaded'
            }
        
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_batch, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            pill_name = self.pill_names[class_idx]
            
            return {
                'pill_name': pill_name,
                'confidence': confidence,
                'method': 'visual_classification'
            }
            
        except Exception as e:
            return {
                'pill_name': 'Unknown',
                'confidence': 0.0,
                'method': 'visual_classification',
                'error': str(e)
            }
    
    # ========================================================================
    # STEP 2: IMPRINT READING (OCR)
    # ========================================================================
    
    def read_imprint(self, image_path):
        """
        Step 2: Read imprint text using OCR
        
        Args:
            image_path: Path to pill image
            
        Returns:
            dict: {imprint_text, dosage, method}
        """
        if not HAS_OCR:
            return {
                'imprint_text': 'OCR not available',
                'dosage': 'Unknown',
                'method': 'ocr',
                'error': 'pytesseract not installed'
            }
        
        try:
            img = Image.open(image_path)
            
            # Convert to grayscale for better OCR
            img_gray = img.convert('L')
            
            # Upscale for better text recognition
            width, height = img_gray.size
            img_upscaled = img_gray.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
            
            # Extract text
            text = pytesseract.image_to_string(img_upscaled)
            text = text.strip()
            
            # Extract dosage
            dosage = self._extract_dosage(text)
            
            return {
                'imprint_text': text,
                'dosage': dosage,
                'method': 'ocr'
            }
            
        except Exception as e:
            return {
                'imprint_text': '',
                'dosage': 'Unknown',
                'method': 'ocr',
                'error': str(e)
            }
    
    def _extract_dosage(self, text):
        """
        Extract dosage from OCR text
        
        Patterns: 500MG, 500 MG, 500mg, 500 mg, 500 milligram
        """
        if not text:
            return 'Unknown'
        
        # Look for dosage patterns
        patterns = [
            r'(\d+)\s*(?:mg|MG|milligram)',
            r'(\d+)\s*(?:mcg|MCG|microgram)',
            r'(\d+)\s*(?:g|G|gram)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1)
                if 'mcg' in text.lower() or 'microgram' in text.lower():
                    return f"{amount} MCG"
                elif 'g' in text.lower() and 'mg' not in text.lower():
                    return f"{amount} G"
                else:
                    return f"{amount} MG"
        
        return 'Unknown'
    
    # ========================================================================
    # STEP 3: DATABASE LOOKUP
    # ========================================================================
    
    def get_metadata_from_db(self, pill_name, dosage):
        """
        Step 3: Lookup pill metadata in database
        
        Args:
            pill_name: Name of pill
            dosage: Dosage (e.g., "500 MG")
            
        Returns:
            dict: {usage, consumption, precaution, side_effects, manufacturer}
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Try exact match first
            query = """
                SELECT usage, consumption, precaution, side_effects, manufacturer
                FROM pills
                WHERE name LIKE ? AND dosage LIKE ?
                LIMIT 1
            """
            
            cursor.execute(query, (f"%{pill_name}%", f"%{dosage.split()[0]}%"))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                return {
                    'usage': result[0],
                    'consumption': result[1],
                    'precaution': result[2],
                    'side_effects': result[3],
                    'manufacturer': result[4]
                }
            else:
                return {
                    'usage': 'No information available',
                    'consumption': 'No information available',
                    'precaution': 'No information available',
                    'side_effects': 'No information available',
                    'manufacturer': 'Unknown'
                }
                
        except Exception as e:
            print(f"Database error: {e}")
            return None
    
    def add_pill_to_db(self, name, dosage, usage, consumption, precaution, side_effects, manufacturer):
        """Add new pill to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO pills (name, dosage, usage, consumption, precaution, side_effects, manufacturer)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, dosage, usage, consumption, precaution, side_effects, manufacturer))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding pill: {e}")
            return False
    
    # ========================================================================
    # MAIN: COMPLETE EXTRACTION PIPELINE
    # ========================================================================
    
    def extract_complete_info(self, image_path):
        """
        Complete Pipeline: Image → Full Pill Information
        
        Returns:
            dict: Complete pill information
        """
        if not os.path.exists(image_path):
            return {'error': f'Image not found: {image_path}'}
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Step 1: Visual Classification
        print("\n[1/3] Visual Classification...")
        visual_info = self.classify_by_image(image_path)
        print(f"  • Pill: {visual_info['pill_name']}")
        print(f"  • Confidence: {visual_info['confidence']:.1%}")
        
        # Step 2: OCR Imprint Reading
        print("\n[2/3] Reading Imprint...")
        ocr_info = self.read_imprint(image_path)
        print(f"  • Imprint: {ocr_info['imprint_text']}")
        print(f"  • Dosage: {ocr_info['dosage']}")
        
        # Step 3: Database Lookup
        print("\n[3/3] Retrieving Metadata...")
        metadata = self.get_metadata_from_db(
            visual_info['pill_name'],
            ocr_info['dosage']
        )
        
        if metadata:
            print(f"  • Usage: {metadata['usage']}")
            print(f"  • Consumption: {metadata['consumption']}")
            print(f"  • Precaution: {metadata['precaution']}")
        else:
            print("  ⚠️ No metadata found in database")
        
        # Compile complete result
        result = {
            'image': os.path.basename(image_path),
            'timestamp': datetime.now().isoformat(),
            'visual_classification': visual_info,
            'imprint_reading': ocr_info,
            'metadata': metadata
        }
        
        return result
    
    def format_result_for_display(self, result):
        """Format result nicely for display"""
        output = f"""
╔{'═'*58}╗
║ {'PILL IDENTIFICATION RESULT':<56} ║
╚{'═'*58}╝

📷 IMAGE: {result['image']}
⏰ TIMESTAMP: {result['timestamp']}

┌─ VISUAL CLASSIFICATION ─────────────────────────────────┐
│ Pill Name:  {result['visual_classification']['pill_name']:<40} │
│ Confidence: {result['visual_classification']['confidence']:>39.1%} │
└─────────────────────────────────────────────────────────┘

┌─ IMPRINT READING ───────────────────────────────────────┐
│ Text: {result['imprint_reading']['imprint_text']:<46} │
│ Dosage: {result['imprint_reading']['dosage']:<45} │
└─────────────────────────────────────────────────────────┘

┌─ PILL INFORMATION ──────────────────────────────────────┐
"""
        
        if result['metadata']:
            output += f"""│ Usage:      {result['metadata']['usage']:<45} │
│ Consumption: {result['metadata']['consumption']:<44} │
│ Precaution: {result['metadata']['precaution']:<45} │
│ Side Effects: {result['metadata']['side_effects']:<44} │
│ Manufacturer: {result['metadata']['manufacturer']:<44} │
"""
        else:
            output += "│ ⚠️ No metadata found                                   │\n"
        
        output += "└─────────────────────────────────────────────────────────┘\n"
        
        return output


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Initialize extractor
    extractor = PillInfoExtractor(
        model_path='media/pilldata/model_feature_learning.keras',
        database_path='pills_database.db'
    )
    
    # Test with sample image
    test_images = [
        'sample_pill_1.jpg',
        'sample_pill_2.jpg',
        'test_pill.jpg'
    ]
    
    # Find first existing image
    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if test_image:
        result = extractor.extract_complete_info(test_image)
        print(extractor.format_result_for_display(result))
        
        # Save result as JSON
        json_output = json.dumps(result, indent=2)
        print(f"\nJSON Output:\n{json_output}")
    else:
        print("No test images found. Usage:")
        print("  python pill_info_extractor.py <image_path>")
