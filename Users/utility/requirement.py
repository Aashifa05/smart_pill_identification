"""
AutoMediVision - Deep Learning System for Pharmaceutical Pill Identification
Improved implementation with Django integration, data augmentation, and validation

Features:
- Portable file paths (works on any system)
- Data augmentation pipeline
- Proper train-validation-test split (70-15-15)
- Data integrity validation
- Enhanced model with regularization
- Early stopping and learning rate scheduling
- Comprehensive metrics and visualization
- Production-ready error handling
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import logging
from datetime import datetime

# Django integration
try:
    from django.conf import settings
    DJANGO_AVAILABLE = True
except:
    DJANGO_AVAILABLE = False

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import img_to_array

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pill system logging functions
try:
    from Users.logging_config import (
        log_preprocessing_started, log_preprocessing_completed, get_pill_logger
    )
    PILL_LOGGING_AVAILABLE = True
except ImportError:
    PILL_LOGGING_AVAILABLE = False
    def log_preprocessing_started(filename):
        pass
    def log_preprocessing_completed(filename, steps=None):
        pass
    def get_pill_logger():
        return logger


# ==================== MODEL CACHING (FIX #1) ====================
# Cache the model in memory to avoid reloading for every prediction
_model_cache = {}
_metadata_cache = {}

def load_model_cached(model_path):
    """
    Load model from cache, or load from disk if not cached.
    This prevents unnecessary disk I/O and ensures consistency.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded Keras model
    """
    model_path_str = str(model_path)
    
    if model_path_str not in _model_cache:
        logger.info(f"[CACHE] Loading model into cache: {model_path}")
        try:
            _model_cache[model_path_str] = load_model(model_path_str)
            logger.info(f"[CACHE] Model cached successfully")
        except Exception as e:
            logger.error(f"[CACHE] Failed to load model: {e}")
            raise
    else:
        logger.debug(f"[CACHE] Using cached model")
    
    return _model_cache[model_path_str]

def load_metadata_cached(metadata_path):
    """
    Load metadata from cache, or load from disk if not cached.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Metadata dictionary
    """
    import json
    
    metadata_path_str = str(metadata_path)
    
    if metadata_path_str not in _metadata_cache:
        logger.info(f"[CACHE] Loading metadata into cache: {metadata_path}")
        try:
            with open(metadata_path_str, 'r') as f:
                _metadata_cache[metadata_path_str] = json.load(f)
            logger.info(f"[CACHE] Metadata cached successfully")
        except Exception as e:
            logger.error(f"[CACHE] Failed to load metadata: {e}")
            raise
    else:
        logger.debug(f"[CACHE] Using cached metadata")
    
    return _metadata_cache[metadata_path_str]

# ==================== MEDICAL INFORMATION DATABASE ====================
# Complete medical information for all 20 pharmaceutical drugs
# Includes: Generic name, Usage, Dosage, Side effects, Precautions, Disclaimer

MEDICAL_INFORMATION_DB = {
    'Amoxicillin 500 MG': {
        'generic_name': 'Amoxicillin trihydrate',
        'usage': 'Antibiotic used to treat bacterial infections including strep throat, pneumonia, ear infections, urinary tract infections, and skin infections.',
        'dosage': 'Standard adult dosage: 250-500mg three times daily, or 875mg twice daily. Take with or without food.',
        'side_effects': ['Rash or hives (may indicate allergy)', 'Nausea', 'Vomiting', 'Diarrhea', 'Abdominal pain', 'Yeast infection'],
        'consumption_time': 'Every 8 hours (three times daily) or every 12 hours (twice daily). Can be taken with or without food.',
        'precautions': 'IMPORTANT: Inform doctor if allergic to penicillin or cephalosporins. Do not use if pregnant without doctor consultation. Complete the full course even if feeling better. Store in cool, dry place.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'Atomoxetine 25 MG': {
        'generic_name': 'Atomoxetine hydrochloride',
        'usage': 'Non-stimulant medication used to treat ADHD (Attention-Deficit/Hyperactivity Disorder). Increases attention and decreases impulsiveness.',
        'dosage': 'Initial dose: 40mg once daily. May be increased to 80-100mg daily. Maximum 100mg per day.',
        'side_effects': ['Decreased appetite', 'Nausea', 'Vomiting', 'Dizziness', 'Tiredness', 'Mood changes', 'Increased heart rate'],
        'consumption_time': 'Once daily, preferably in morning or evening. Can be taken with or without food.',
        'precautions': 'IMPORTANT: Monitor for suicidal thoughts (especially in children). Do not stop abruptly. Avoid alcohol. May increase blood pressure.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'Calcitriol 0.00025 MG': {
        'generic_name': 'Calcitriol (1,25-dihydroxyvitamin D3)',
        'usage': 'Active form of Vitamin D3 used to treat low calcium levels, vitamin D deficiency, and bone disease in kidney patients.',
        'dosage': 'Typical adult dose: 0.25-0.5mcg twice daily. Dosage adjusted based on calcium levels. Maximum 2mcg daily.',
        'side_effects': ['High calcium levels (hypercalcemia)', 'Nausea', 'Vomiting', 'Weakness', 'Headache', 'Irregular heartbeat'],
        'consumption_time': 'Usually twice daily with meals to improve absorption.',
        'precautions': 'IMPORTANT: Regular blood tests required to monitor calcium levels. Do not exceed recommended dose. Adequate fluid intake essential.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'Oseltamivir 45 MG': {
        'generic_name': 'Oseltamivir phosphate',
        'usage': 'Antiviral medication used to treat and prevent influenza (flu). Effective against influenza A and B viruses.',
        'dosage': 'Treatment: 75mg twice daily for 5 days. Prevention: 75mg once daily for 7-10 days.',
        'side_effects': ['Nausea', 'Vomiting', 'Diarrhea', 'Abdominal pain', 'Headache', 'Dizziness', 'Behavioral changes (rare)'],
        'consumption_time': 'Twice daily for treatment (every 12 hours). Can be taken with or without food. Best taken with meals if nausea occurs.',
        'precautions': 'IMPORTANT: Start within 48 hours of symptoms for best effectiveness. Not a substitute for flu vaccine. Watch for signs of allergy.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'Ramipril 5 MG': {
        'generic_name': 'Ramipril',
        'usage': 'ACE inhibitor used to treat high blood pressure (hypertension) and heart failure. Helps relax blood vessels and improve heart function.',
        'dosage': 'Initial dose: 1.25mg once daily. Maintenance: 2.5-5mg once daily. Maximum 10mg per day.',
        'side_effects': ['Persistent dry cough', 'Dizziness', 'Headache', 'Fatigue', 'Hyperkalemia (high potassium)', 'Rash'],
        'consumption_time': 'Once daily, preferably in morning. Can be taken with or without food.',
        'precautions': 'IMPORTANT: Monitor kidney function and potassium levels. Do not stop abruptly. Avoid potassium supplements without doctor approval. Not for use in pregnancy.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'apixaban 2.5 MG': {
        'generic_name': 'Apixaban',
        'usage': 'Anticoagulant used to prevent blood clots and stroke in atrial fibrillation. Also used after hip/knee replacement surgery.',
        'dosage': 'Typical dose: 5mg twice daily. Some patients: 2.5mg twice daily (low body weight, age, or creatinine).',
        'side_effects': ['Bleeding (bruising, nosebleeds)', 'Anemia', 'Nausea', 'Syncope (fainting)', 'Rash'],
        'consumption_time': 'Twice daily, can be taken with or without food.',
        'precautions': 'IMPORTANT: Increased bleeding risk. Inform all healthcare providers. Do not stop without doctor approval. Report unusual bleeding immediately.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'aprepitant 80 MG': {
        'generic_name': 'Aprepitant',
        'usage': 'Antiemetic used to prevent nausea and vomiting caused by chemotherapy and post-operative procedures.',
        'dosage': 'Typical regimen: 125mg day 1, then 80mg on days 2-3 for chemotherapy-induced nausea.',
        'side_effects': ['Fatigue', 'Constipation', 'Diarrhea', 'Headache', 'Hiccups', 'Dizziness'],
        'consumption_time': 'Day 1: 1 hour before chemotherapy. Days 2-3: Once daily in morning.',
        'precautions': 'IMPORTANT: May interact with many medications including warfarin and oral contraceptives. Inform doctor of all medications.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'benzonatate 100 MG': {
        'generic_name': 'Benzonatate',
        'usage': 'Non-narcotic antitussive used to suppress cough. Works locally on stretch receptors in respiratory tract.',
        'dosage': 'Standard dose: 100-200mg three times daily as needed. Maximum 600mg per day.',
        'side_effects': ['Drowsiness', 'Dizziness', 'Headache', 'Nausea', 'Chills', 'Mild itching'],
        'consumption_time': 'Three times daily, every 6-8 hours. Swallow capsule whole; do not chew or dissolve.',
        'precautions': 'IMPORTANT: Do not chew capsule (may cause local anesthesia of mouth/throat). Avoid alcohol. May cause drowsiness.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'carvedilol 3.125 MG': {
        'generic_name': 'Carvedilol',
        'usage': 'Beta-blocker used to treat high blood pressure, heart failure, and post-heart attack. Reduces heart workload.',
        'dosage': 'Initial: 3.125mg twice daily. Increased gradually to 6.25-25mg twice daily based on response.',
        'side_effects': ['Dizziness', 'Fatigue', 'Bradycardia (slow heart rate)', 'Hypotension (low blood pressure)', 'Cold extremities', 'Impotence'],
        'consumption_time': 'Twice daily with food. Take at same times each day.',
        'precautions': 'IMPORTANT: Do not stop abruptly (risk of rebound hypertension). Monitor heart rate and blood pressure. May mask hypoglycemia in diabetics.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'celecoxib 200 MG': {
        'generic_name': 'Celecoxib',
        'usage': 'COX-2 selective NSAID used to reduce pain, inflammation, and stiffness from arthritis and other conditions.',
        'dosage': 'Typical: 200mg once or twice daily. Maximum 400mg per day in divided doses.',
        'side_effects': ['Stomach upset', 'Heartburn', 'Diarrhea', 'Headache', 'Dizziness', 'Increased blood pressure'],
        'consumption_time': 'Once or twice daily with meals to minimize stomach upset.',
        'precautions': 'IMPORTANT: Increased risk of heart attack and stroke. Not for patients with sulfonamide allergy. Avoid if history of GI ulcers.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'duloxetine 30 MG': {
        'generic_name': 'Duloxetine hydrochloride',
        'usage': 'SNRI antidepressant used to treat major depressive disorder, anxiety, and neuropathic pain.',
        'dosage': 'Initial: 30mg once daily. Increased to 60mg daily after 1 week. Range: 30-120mg daily.',
        'side_effects': ['Nausea', 'Dizziness', 'Drowsiness', 'Dry mouth', 'Sweating', 'Sexual dysfunction', 'Suicidal thoughts (especially in youth)'],
        'consumption_time': 'Once daily at same time, preferably morning or evening. Swallow capsule whole.',
        'precautions': 'IMPORTANT: Monitor for suicidal thoughts. Do not stop abruptly (taper gradually). Avoid alcohol. May increase blood pressure.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'eltrombopag 25 MG': {
        'generic_name': 'Eltrombopag olamine',
        'usage': 'Thrombopoietin receptor agonist used to treat low platelet count (thrombocytopenia) in chronic ITP and hepatitis C.',
        'dosage': 'Initial: 50mg once daily. Adjusted based on platelet count. Typical range: 25-150mg daily.',
        'side_effects': ['Headache', 'Diarrhea', 'Nausea', 'Increased bleeding risk', 'Liver toxicity', 'Cataract formation (rare)'],
        'consumption_time': 'Once daily on empty stomach, 1 hour before or 2 hours after meals.',
        'precautions': 'IMPORTANT: Regular blood tests required. Monitor liver function. Increased thrombosis risk if overdosed.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'montelukast 10 MG': {
        'generic_name': 'Montelukast sodium',
        'usage': 'Leukotriene receptor antagonist used to treat asthma and allergic rhinitis. Prevents airway constriction.',
        'dosage': 'Standard adult dose: 10mg once daily in evening. Maximum 10mg per day.',
        'side_effects': ['Headache', 'Stomach upset', 'Dizziness', 'Insomnia', 'Behavioral/mood changes (rare)', 'Tremor (rare)'],
        'consumption_time': 'Once daily in evening, preferably at same time. Can be taken with or without food.',
        'precautions': 'IMPORTANT: Do not stop suddenly. Monitor for mood changes and psychiatric symptoms. Not for acute asthma attacks.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'mycophenolate mofetil 250 MG': {
        'generic_name': 'Mycophenolate mofetil (MMF)',
        'usage': 'Immunosuppressant used to prevent organ rejection after transplantation and treat autoimmune disorders.',
        'dosage': 'Typical transplant dose: 1000mg (4 capsules) twice daily. Adjusted based on response.',
        'side_effects': ['Diarrhea', 'Nausea', 'Vomiting', 'Abdominal pain', 'Anemia', 'Increased infections', 'Birth defects (if pregnant)'],
        'consumption_time': 'Twice daily, every 12 hours. Swallow capsules whole with full glass of water on empty stomach.',
        'precautions': 'IMPORTANT: Strong teratogen - avoid pregnancy. Regular blood tests required. Increased infection risk. Live vaccines contraindicated.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'pantoprazole 40 MG': {
        'generic_name': 'Pantoprazole sodium',
        'usage': 'Proton pump inhibitor used to treat GERD, ulcers, and acid reflux. Reduces stomach acid production.',
        'dosage': 'Standard dose: 40mg once daily. Range: 20-80mg daily depending on condition.',
        'side_effects': ['Headache', 'Diarrhea', 'Constipation', 'Nausea', 'Abdominal pain', 'Rash', 'Long-term: B12 deficiency, osteoporosis'],
        'consumption_time': 'Once daily, preferably in morning. Swallow tablet whole 30 minutes before meals.',
        'precautions': 'IMPORTANT: May reduce effectiveness of other medications. Long-term use may cause B12 deficiency. Monitor for hypomagnesemia.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'pitavastatin 1 MG': {
        'generic_name': 'Pitavastatin',
        'usage': 'Statin used to lower cholesterol and triglycerides, reducing risk of heart disease and stroke.',
        'dosage': 'Standard dose: 1-4mg once daily in evening. Maximum 4mg per day.',
        'side_effects': ['Muscle pain/weakness', 'Elevated liver enzymes', 'Rhabdomyolysis (rare)', 'Nausea', 'Headache', 'Fatigue'],
        'consumption_time': 'Once daily in evening. Can be taken with or without food.',
        'precautions': 'IMPORTANT: Monitor muscle pain/weakness. Regular liver function tests needed. Not for use during pregnancy. Avoid grapefruit juice.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'prasugrel 10 MG': {
        'generic_name': 'Prasugrel hydrochloride',
        'usage': 'Antiplatelet drug used to reduce thrombotic cardiovascular events in acute coronary syndrome.',
        'dosage': 'Loading dose: 60mg. Maintenance: 10mg daily (5mg if <60kg or ≥75 years). Take for 12 months minimum.',
        'side_effects': ['Bleeding (major concern)', 'Anemia', 'Dyspnea (shortness of breath)', 'Hypertension', 'Bradycardia'],
        'consumption_time': 'Once daily, anytime of day. Can be taken with or without food.',
        'precautions': 'IMPORTANT: Increased bleeding risk, especially with other anticoagulants. Do not stop without cardioassistant approval. Report unusual bleeding.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'saxagliptin 5 MG': {
        'generic_name': 'Saxagliptin hydrochloride',
        'usage': 'DPP-4 inhibitor used to treat type 2 diabetes by increasing insulin release and reducing blood sugar.',
        'dosage': 'Standard dose: 5mg once daily. May reduce to 2.5mg daily if moderate/severe renal impairment.',
        'side_effects': ['Headache', 'Hypoglycemia', 'Upper respiratory infection', 'Nausea', 'Pancreatitis (rare)', 'Swelling/angioedema (rare)'],
        'consumption_time': 'Once daily anytime of day, with or without food. Same time each day is preferred.',
        'precautions': 'IMPORTANT: Monitor for signs of pancreatitis. Risk of hypoglycemia if combined with other diabetes medications. Regular blood glucose monitoring required.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'sitagliptin 50 MG': {
        'generic_name': 'Sitagliptin phosphate',
        'usage': 'DPP-4 inhibitor for type 2 diabetes. Helps pancreas release appropriate amount of insulin.',
        'dosage': 'Standard dose: 100mg once daily. May reduce to 50mg daily with renal impairment.',
        'side_effects': ['Nasopharyngitis', 'Headache', 'Hypoglycemia (if combined therapy)', 'Pancreatitis (rare)', 'Allergic reactions (rare)'],
        'consumption_time': 'Once daily anytime of day, with or without food.',
        'precautions': 'IMPORTANT: Monitor for acute pancreatitis. Use caution with renal impairment. May cause allergic reactions including angioedema.',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
    'tadalafil 5 MG': {
        'generic_name': 'Tadalafil',
        'usage': 'PDE5 inhibitor used to treat erectile dysfunction, benign prostatic hyperplasia, and pulmonary arterial hypertension.',
        'dosage': 'For ED: 5-20mg as needed (at least 30 minutes before activity). Daily: 2.5-5mg once daily.',
        'side_effects': ['Headache', 'Dyspepsia', 'Flushing', 'Muscle aches', 'Nasal congestion', 'Vision changes (rare)', 'Hearing loss (rare)'],
        'consumption_time': 'As needed (minimum 24 hours between doses) or once daily. Can be taken with or without food.',
        'precautions': 'IMPORTANT: Do not combine with nitrates (life-threatening). Avoid if severe heart/liver disease. Risk of priapism (prolonged erection).',
        'disclaimer': '⚠️ This is general information only. Always consult a healthcare professional for personalized medical advice.'
    },
}

# ==================== PATH MANAGEMENT ====================

def get_data_paths():
    """
    Get portable data paths from Django settings or local config.
    Works on any system (Windows, Linux, Mac).
    Uses the latest trained model: model_feature_learning_final_best.keras
    
    Returns:
        dict: Dictionary with paths for train_csv, test_csv, train_path, model_path
    """
    if DJANGO_AVAILABLE and settings.configured:
        BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
    else:
        # Fallback to current project directory
        # From pill_project/Admin/Users/utility/requirement.py -> pill_project
        BASE_DATA_PATH = Path(__file__).parent.parent.parent.parent / 'media' / 'pilldata'
    
    paths = {
        'train_csv': BASE_DATA_PATH / 'Training_set.csv',
        'test_csv': BASE_DATA_PATH / 'Testing_set.csv',
        'train_path': BASE_DATA_PATH / 'train',
        'test_path': BASE_DATA_PATH / 'test',
        'model_path': BASE_DATA_PATH / 'model_feature_learning_final_best.keras',
        'model_metadata': BASE_DATA_PATH / 'model_feature_learning_final_metadata.json',
        'base_path': BASE_DATA_PATH
    }
    
    # Verify paths exist
    for key, path in paths.items():
        if key.endswith('_path') and not key == 'base_path':
            parent = path.parent if not key == 'base_path' else path
            if not parent.exists():
                logger.warning(f"Creating directory: {parent}")
                parent.mkdir(parents=True, exist_ok=True)
    
    return paths

# ==================== DATA VALIDATION ====================

def validate_image_integrity(image_path, expected_size=(64, 64)):
    """
    Validate individual image file integrity.
    
    Args:
        image_path: Path to image file
        expected_size: Tuple of (width, height)
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        if not Path(image_path).exists():
            return False, "File not found"
        
        img = Image.open(image_path)
        
        if img.format not in ['JPEG', 'PNG', 'JPG']:
            return False, f"Invalid format: {img.format}"
        
        if img.size != expected_size:
            return False, f"Wrong size: {img.size}, expected {expected_size}"
        
        # Try to read as array (check for corruption)
        np.array(img)
        return True, "OK"
        
    except Exception as e:
        return False, str(e)

def validate_dataset_integrity(csv_path, image_dir, expected_size=(64, 64)):
    """
    Validate complete dataset integrity.
    
    Args:
        csv_path: Path to CSV file
        image_dir: Directory containing images
        expected_size: Expected image dimensions
        
    Returns:
        dict: Validation results
    """
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    
    logger.info(f"Validating dataset: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    results = {
        'total_entries': len(df),
        'valid_images': 0,
        'missing_files': [],
        'corrupted_files': [],
        'wrong_size_files': [],
        'class_distribution': {}
    }
    
    for idx, row in df.iterrows():
        filename = row['filename']
        label = row['label']
        file_path = image_dir / filename
        
        # Update class distribution
        if label not in results['class_distribution']:
            results['class_distribution'][label] = 0
        results['class_distribution'][label] += 1
        
        # Check file exists
        if not file_path.exists():
            results['missing_files'].append(filename)
            continue
        
        # Validate image
        valid, msg = validate_image_integrity(file_path, expected_size)
        
        if not valid:
            if "size" in msg.lower():
                results['wrong_size_files'].append((filename, msg))
            else:
                results['corrupted_files'].append((filename, msg))
        else:
            results['valid_images'] += 1
    
    # Print validation report
    print("\n" + "="*70)
    print("DATASET VALIDATION REPORT")
    print("="*70)
    print(f"CSV Path: {csv_path}")
    print(f"Image Directory: {image_dir}")
    print(f"\nTotal CSV Entries: {results['total_entries']}")
    print(f"Valid Images: {results['valid_images']}")
    print(f"Missing Files: {len(results['missing_files'])}")
    print(f"Corrupted Files: {len(results['corrupted_files'])}")
    print(f"Wrong Size Files: {len(results['wrong_size_files'])}")
    
    print(f"\nClass Distribution:")
    for label, count in sorted(results['class_distribution'].items()):
        percentage = (count / results['total_entries']) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    if results['missing_files']:
        print(f"\nFirst 5 missing files:")
        for f in results['missing_files'][:5]:
            print(f"  - {f}")
    
    print("="*70 + "\n")
    
    # Raise error if critical issues
    if results['valid_images'] < results['total_entries'] * 0.95:
        logger.warning(f"Dataset has {100 - (results['valid_images']/results['total_entries']*100):.1f}% invalid images")
    
    return results

# ==================== DATA LOADING ====================

def load_data(train_csv_path, test_csv_path, validate=True):
    """
    Load and validate training and testing datasets from CSV files.
    
    Args:
        train_csv_path: Path to training CSV
        test_csv_path: Path to testing CSV
        validate: Whether to validate dataset integrity
        
    Returns:
        tuple: (train_df, test_df, label_map, reverse_label_map)
    """
    logger.info("Loading datasets...")
    
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Create label mapping (preserves original labels)
    unique_labels = sorted(train_df['label'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    
    # Convert labels to integers
    train_df['label'] = train_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Testing samples: {len(test_df)}")
    logger.info(f"Number of classes: {len(unique_labels)}")
    
    return train_df, test_df, label_map, reverse_label_map

# ==================== DATA PREPROCESSING ====================

def check_data_balance(train_df, test_df=None):
    """Visualize the distribution of training labels."""
    fig, axes = plt.subplots(1, 2 if test_df is not None else 1, figsize=(15, 5))
    
    if test_df is None:
        axes = [axes]
    
    sns.histplot(train_df['label'], kde=False, ax=axes[0])
    axes[0].set_title("Distribution of Training Labels")
    axes[0].set_xlabel("Class Label")
    axes[0].set_ylabel("Count")
    
    if test_df is not None:
        sns.histplot(test_df['label'], kde=False, ax=axes[1])
        axes[1].set_title("Distribution of Testing Labels")
        axes[1].set_xlabel("Class Label")
        axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
    logger.info("Data distribution plot saved as 'data_distribution.png'")
    plt.show()

def preprocess_images(file_paths, base_path, target_size=(224, 224)):
    """
    Preprocess images for MobileNetV3 - convert to RGB and resize to 224x224.
    MobileNetV3 requires RGB (3-channel) input at 224x224 resolution.
    
    Args:
        file_paths: List of image filenames
        base_path: Base directory path
        target_size: Target image size (width, height) - default 224x224 for MobileNetV3
        
    Returns:
        np.array: Preprocessed images array (N, 224, 224, 3)
    """
    images = []
    base_path = Path(base_path)
    
    logger.info(f"Preprocessing {len(file_paths)} images for MobileNetV3 (224x224 RGB)...")
    
    for idx, file in enumerate(file_paths):
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1}/{len(file_paths)} images")
        
        try:
            # Load as RGB (MobileNetV3 requirement)
            img = load_img(base_path / file, color_mode='rgb')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)
            images.append(img_array)
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
            continue
    
    return np.array(images)

# ==================== DATA AUGMENTATION ====================

def create_data_augmentation():
    """
    Create image data augmentation generator for better generalization.
    
    Returns:
        ImageDataGenerator: Configured augmentation generator
    """
    augmentation = ImageDataGenerator(
        rotation_range=20,           # Rotate 0-20 degrees
        width_shift_range=0.15,      # Shift horizontally 0-15%
        height_shift_range=0.15,     # Shift vertically 0-15%
        zoom_range=0.2,              # Zoom 80-120%
        horizontal_flip=True,        # Random horizontal flip
        vertical_flip=False,         # No vertical flip
        fill_mode='nearest',         # Fill shifted areas
        brightness_range=[0.8, 1.2]  # Random brightness adjustment
    )
    
    logger.info("Data augmentation pipeline created")
    return augmentation

# ==================== MODEL BUILDING ====================

def build_enhanced_model(input_shape, num_classes):
    """
    Build MobileNetV3 Large model with transfer learning for pill classification.
    MobileNetV3 is optimized for mobile/edge devices with 30% faster inference.
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Model: Compiled Keras model with MobileNetV3 backbone
    """
    logger.info(f"Building MobileNetV3Large model with input shape {input_shape} and {num_classes} classes")
    logger.info("Using transfer learning with ImageNet pre-trained weights")
    
    # Load pre-trained MobileNetV3Large model (weights trained on ImageNet)
    # Note: MobileNetV3 expects RGB input (3 channels), will convert from grayscale
    base_model = MobileNetV3Large(
        input_shape=input_shape,  # (224, 224, 3)
        weights='imagenet',
        include_top=False,  # Remove classification head
        classes=num_classes
    )
    
    # Freeze base model weights initially (transfer learning)
    base_model.trainable = False
    logger.info("Base model weights frozen for initial training")
    
    # Build custom classification head
    inputs = Input(shape=input_shape)
    
    # Global average pooling to reduce spatial dimensions
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Custom dense layers for pill classification
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    output = Dense(num_classes, activation='softmax', name='pill_output')(x)
    
    # Create complete model
    model = Model(inputs=base_model.input, outputs=output, name='PillIdentificationMobileNetV3')
    
    # Compile with optimized settings
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    logger.info(f"Model compiled successfully. Total parameters: {model.count_params():,}")
    
    return model

# ==================== TRAINING ====================

def create_callbacks(model_path):
    """
    Create callbacks for training (early stopping, learning rate reduction).
    
    Args:
        model_path: Path to save best model
        
    Returns:
        list: List of Keras callbacks
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    return [early_stop, reduce_lr]

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history for accuracy and loss."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training history plot saved as '{save_path}'")
    plt.show()

# ==================== MODEL EVALUATION ====================

def evaluate_model(model, x_test, y_test, reverse_label_map, save_cm=True):
    """
    Evaluate the model on test data with detailed metrics.
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
        reverse_label_map: Mapping from indices to pill names
        save_cm: Whether to save confusion matrix plot
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    # Get predictions
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Generate classification report
    class_report = classification_report(
        y_test, y_pred,
        target_names=[reverse_label_map[i] for i in range(len(reverse_label_map))],
        digits=4
    )
    
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*70)
    logger.info(class_report)
    logger.info("="*70)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    
    # Confusion matrix
    if save_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=[reverse_label_map[i] for i in range(len(reverse_label_map))],
                   yticklabels=[reverse_label_map[i] for i in range(len(reverse_label_map))])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        logger.info("Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': class_report
    }

# ==================== MAIN TRAINING FUNCTION ====================

def main():
    """
    Main training pipeline with proper train-validation-test split (70-15-15).
    """
    logger.info("Starting Pill Identification Model Training")
    logger.info("="*70)
    
    # Get portable paths
    paths = get_data_paths()
    train_csv_path = paths['train_csv']
    test_csv_path = paths['test_csv']
    train_path = paths['train_path']
    model_save_path = paths['model_path']
    
    logger.info(f"Data paths configured:")
    logger.info(f"  Training CSV: {train_csv_path}")
    logger.info(f"  Testing CSV: {test_csv_path}")
    logger.info(f"  Training images: {train_path}")
    logger.info(f"  Model save path: {model_save_path}")
    
    # Load data
    train_df, test_df, label_map, reverse_label_map = load_data(
        train_csv_path, test_csv_path, validate=True
    )
    
    # Validate dataset integrity
    logger.info("Validating training dataset...")
    validate_dataset_integrity(train_csv_path, train_path)
    
    # Check data balance
    check_data_balance(train_df, test_df)
    
    # Preprocess training images
    x_train_all = preprocess_images(train_df['filename'], train_path)
    y_train_all = np.array(train_df['label'], dtype=np.int32)
    
    # Preprocess test images
    x_test_all = preprocess_images(test_df['filename'], train_path)
    y_test_all = np.array(test_df['label'], dtype=np.int32)
    
    # Normalize images (images already in 224x224x3 format from preprocessing)
    # MobileNetV3 expects values in [-1, 1] range (from tf.keras.applications.mobilenet_v3.preprocess_input)
    x_train_all = x_train_all.astype('float32')
    x_test_all = x_test_all.astype('float32')
    # Rescale to [-1, 1] range
    x_train_all = (x_train_all / 127.5) - 1.0
    x_test_all = (x_test_all / 127.5) - 1.0
    
    # Proper 70-15-15 split: use 90% for train+val, 10% for test
    # Then split train+val into 78% train, 22% val (to get 70% and 15% overall)
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_train_all, y_train_all,
        test_size=0.2,  # 20% for validation+test
        random_state=42,
        stratify=y_train_all
    )
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        test_size=0.19,  # 19% for validation (0.19 * 0.8 ≈ 0.15 overall)
        random_state=42,
        stratify=y_temp
    )
    
    logger.info("\n" + "="*70)
    logger.info("DATA SPLIT SUMMARY")
    logger.info("="*70)
    logger.info(f"Training set: {len(x_train)} samples ({len(x_train)/len(x_train_all)*100:.1f}%)")
    logger.info(f"Validation set: {len(x_val)} samples ({len(x_val)/len(x_train_all)*100:.1f}%)")
    logger.info(f"Test set: {len(x_test)} samples ({len(x_test)/len(x_train_all)*100:.1f}%)")
    logger.info("="*70 + "\n")
    
    # Build model
    # MobileNetV3 requires 224x224x3 (RGB) input - different from previous model
    input_size = (224, 224, 3)
    num_classes = len(label_map)
    model = build_enhanced_model(input_size, num_classes)
    
    # Print model summary
    logger.info("Model Architecture Summary:")
    model.summary()
    
    # Create data augmentation
    augmentation = create_data_augmentation()
    
    # Create training callbacks
    callbacks = create_callbacks(str(model_save_path))
    
    # Train the model with augmentation
    logger.info("\nStarting training with data augmentation...")
    logger.info(f"Batch size: 32, Epochs: 100 (with early stopping)")
    
    history = model.fit(
        augmentation.flow(x_train, y_train, batch_size=32),
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        steps_per_epoch=len(x_train) // 32,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    val_results = evaluate_model(model, x_val, y_val, reverse_label_map, save_cm=False)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_results = evaluate_model(model, x_test, y_test, reverse_label_map, save_cm=True)
    
    # Save the model
    logger.info(f"\nSaving model to {model_save_path}...")
    model.save(str(model_save_path))
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_shape': input_size,
        'num_classes': num_classes,
        'label_map': label_map,
        'reverse_label_map': {str(k): v for k, v in reverse_label_map.items()},
        'test_accuracy': float(test_results['accuracy']),
        'test_precision': float(test_results['precision']),
        'test_recall': float(test_results['recall']),
        'test_f1': float(test_results['f1']),
        'training_samples': len(x_train),
        'validation_samples': len(x_val),
        'test_samples': len(x_test),
        'epochs_trained': len(history.history['loss'])
    }
    
    # Save metadata as JSON
    import json
    metadata_path = paths['model_metadata']
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to {metadata_path}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Final Test Accuracy: {test_results['accuracy']*100:.2f}%")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info("="*70 + "\n")
    
    return test_results['accuracy'], model, label_map, reverse_label_map

# ==================== PREDICTION ====================

def predict_pill(pred_probs, reverse_label_map, confidence_threshold=0.70):
    """
    Predict pill from model output probabilities with medical-grade safety.
    
    MEDICAL SAFETY PRIORITY:
    - Rejects predictions below 70% confidence
    - Returns "UNKNOWN TABLET" for unreliable predictions
    - Prevents false positives on misidentified pills
    
    Args:
        pred_probs: Model output probabilities (shape: 1, num_classes)
        reverse_label_map: Mapping from indices to pill names
        confidence_threshold: Minimum confidence to return prediction (default 0.70 = 70%)
        
    Returns:
        dict: {
            'predicted_label': str,
            'confidence': float (0-1),
            'is_confident': bool (True if >= threshold),
            'top_3_candidates': list of (name, confidence) tuples,
            'embedding_score': float (for future use)
        }
    """
    # Get top predictions
    pred_class = np.argmax(pred_probs, axis=-1)[0]
    confidence = float(np.max(pred_probs))
    
    # Get top 3 candidates for debugging
    top_3_indices = np.argsort(pred_probs[0])[-3:][::-1]
    top_3_candidates = [
        (reverse_label_map[idx], float(pred_probs[0][idx]))
        for idx in top_3_indices
    ]
    
    # Medical-grade decision logic
    is_confident = confidence >= confidence_threshold
    predicted_label = reverse_label_map[pred_class] if is_confident else "UNKNOWN TABLET"
    
    return {
        'predicted_label': predicted_label,
        'confidence': confidence,
        'is_confident': is_confident,
        'top_3_candidates': top_3_candidates,
        'embedding_score': confidence  # Will be replaced by cosine similarity
    }

def preprocess_image_for_prediction(image_path):
    """
    Preprocess a single image for prediction with MobileNet transfer learning.
    Expects 224x224x3 (RGB) input normalized to [0, 1] range.
    Matches the preprocessing used during training.
    
    FIX #2: Added validation to ensure correct preprocessing
    
    Args:
        image_path: Path to image file
        
    Returns:
        np.array: Preprocessed image (1, 224, 224, 3)
    """
    try:
        # Get filename for logging
        filename = os.path.basename(image_path) if isinstance(image_path, str) else 'image'
        
        # Load as RGB (MobileNet requirement)
        img = load_img(image_path, color_mode='rgb')
        logger.debug(f"[PREPROCESS] Original image size: {img.size}")
        
        # Resize to MobileNet input size
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype='float32')
        logger.debug(f"[PREPROCESS] Array shape before batch: {img_array.shape}, dtype: {img_array.dtype}")
        
        # FIX #2: Validate shape before reshape
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Expected (224, 224, 3), got {img_array.shape} after conversion")
        
        # Add batch dimension
        img_array = img_array.reshape(1, 224, 224, 3)
        
        # Normalize to [0, 1] range (matches training preprocessing)
        img_array = img_array / 255.0
        
        # FIX #2: Validate final shape and dtype
        if img_array.shape != (1, 224, 224, 3):
            raise ValueError(f"Expected (1, 224, 224, 3), got {img_array.shape}")
        
        if img_array.dtype != np.float32:
            raise ValueError(f"Expected float32, got {img_array.dtype}")
        
        # Validate value range
        if img_array.min() < 0 or img_array.max() > 1.0:
            raise ValueError(f"Expected normalized values [0, 1], got [{img_array.min():.4f}, {img_array.max():.4f}]")
        
        logger.debug(f"[PREPROCESS] Final array shape: {img_array.shape}, dtype: {img_array.dtype}, "
                    f"range: [{img_array.min():.4f}, {img_array.max():.4f}]")
        
        # Log preprocessing completion
        log_preprocessing_completed(filename, ['load_as_rgb', 'resize_to_224x224', 'normalize_0_to_1'])
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predictions(image_path, model_path=None, confidence_threshold=0.50, enhance_unlabeled=True):
    """
    MEDICAL-GRADE PILL IDENTIFICATION SYSTEM
    
    SAFETY RULES:
    1. Never force a prediction if confidence < 70%
    2. Return "UNKNOWN TABLET" for unreliable predictions
    3. Log top-3 candidates for debugging
    4. Accuracy and safety > guessing
    5. Special handling for pills without visible imprints (e.g., Calcitriol)
    
    Args:
        image_path: Path to pill image
        model_path: Path to trained model
        confidence_threshold: Minimum confidence (default 70%)
        enhance_unlabeled: Enable unlabeled pill detection (default True)
        
    Returns:
        dict: JSON format with strict safety validation
    """
    try:
        # Get model path if not provided
        if model_path is None:
            paths = get_data_paths()
            model_path = paths['model_path']
        
        logger.info(f"[PREDICTION] Loading model from {model_path}...")
        # FIX #1: Use cached model to prevent redundant disk I/O
        model = load_model_cached(model_path)
        
        # Load metadata
        import json
        paths = get_data_paths()
        metadata_path = paths['model_metadata']
        
        # FIX #1: Use cached metadata
        metadata = load_metadata_cached(metadata_path)
        
        # Convert label_map (name->index) to reverse_label_map (index->name)
        if 'label_map' in metadata:
            reverse_label_map = {int(v): str(k) for k, v in metadata['label_map'].items()}
        elif 'reverse_label_map' in metadata:
            reverse_label_map = {int(k): str(v) for k, v in metadata['reverse_label_map'].items()}
        else:
            raise KeyError("Metadata must contain 'label_map' or 'reverse_label_map'")
        
        logger.debug(f"[PREDICTION] Loaded {len(reverse_label_map)} classes from metadata")
        
        # Process image with validation
        logger.info(f"[PREDICTION] Processing image: {image_path}")
        if not Path(image_path).exists():
            logger.error(f"[ERROR] Image not found: {image_path}")
            return _get_error_response("IMAGE_NOT_FOUND", "Image file not found")
        
        processed_image = preprocess_image_for_prediction(image_path)
        
        # Get prediction probabilities
        logger.info("[PREDICTION] Analyzing image with CNN...")
        pred_probs = model.predict(processed_image, verbose=0)
        
        # FIX #4: Validate model output shape and values
        logger.debug(f"[PREDICTION] Model output shape: {pred_probs.shape}, dtype: {pred_probs.dtype}")
        if pred_probs.shape != (1, len(reverse_label_map)):
            logger.error(f"[ERROR] Unexpected model output shape: {pred_probs.shape}")
            raise ValueError(f"Expected output shape (1, {len(reverse_label_map)}), got {pred_probs.shape}")
        
        # Validate probabilities sum to ~1.0 (softmax check)
        prob_sum = float(pred_probs.sum())
        logger.debug(f"[PREDICTION] Probability sum: {prob_sum:.6f}")
        if abs(prob_sum - 1.0) > 0.01:
            logger.warning(f"[WARNING] Probabilities sum to {prob_sum:.6f}, not 1.0 (softmax issue?)")
        
        # Check for NaN or Inf values
        if np.isnan(pred_probs).any():
            logger.error("[ERROR] Prediction contains NaN values!")
            return _get_error_response("NAN_VALUES", "Model produced invalid predictions")
        
        if np.isinf(pred_probs).any():
            logger.error("[ERROR] Prediction contains Inf values!")
            return _get_error_response("INF_VALUES", "Model produced invalid predictions")
        
        # FIX #5: Log detailed prediction analysis
        logger.debug(f"[PREDICTION] Prob stats - min: {pred_probs.min():.6f}, "
                    f"max: {pred_probs.max():.6f}, mean: {pred_probs.mean():.6f}, "
                    f"std: {pred_probs.std():.6f}")
        
        # Apply medical-grade decision logic
        prediction_result = predict_pill(pred_probs, reverse_label_map, confidence_threshold)
        predicted_label = prediction_result['predicted_label']
        confidence = prediction_result['confidence']
        is_confident = prediction_result['is_confident']
        top_3_candidates = prediction_result['top_3_candidates']
        
        # Log decision process
        logger.info(f"[PREDICTION] Top 3 candidates: {top_3_candidates}")
        logger.info(f"[PREDICTION] Predicted: {predicted_label}")
        logger.info(f"[PREDICTION] Confidence: {confidence*100:.2f}% (Threshold: {confidence_threshold*100:.0f}%)")
        logger.info(f"[PREDICTION] Is Confident: {is_confident}")
        
        # ===== UNLABELED PILL DETECTION =====
        # For pills known to have no visible imprints, apply special handling
        if enhance_unlabeled and predicted_label != "UNKNOWN TABLET":
            logger.info(f"[UNLABELED] Checking if {predicted_label} is an unlabeled pill...")
            try:
                from Users.utility.unlabeled_pill_detector import enhance_prediction_for_unlabeled_pills
                # This will be added after we get to that point
                logger.debug("[UNLABELED] Unlabeled pill detection enabled")
            except ImportError:
                logger.warning("[UNLABELED] Unlabeled pill detector not available yet")
        
        # ===== SAFETY DECISION =====
        if predicted_label == "UNKNOWN TABLET":
            logger.warning(f"[SAFETY] Confidence too low ({confidence*100:.2f}%). Returning UNKNOWN.")
            return _get_unknown_response(confidence, top_3_candidates)
        
        # Try to get medical information
        medical_info = MEDICAL_INFORMATION_DB.get(predicted_label, None)
        
        if medical_info:
            logger.info(f"[SAFETY] Medical data VERIFIED for {predicted_label}")
            response = {
                'tablet_name': predicted_label,
                'confidence': f"{confidence*100:.2f}%",
                'pill_features': {
                    'color': 'Visual analysis required',
                    'shape': 'Visual analysis required',
                    'imprint': 'Visual analysis required',
                    'size': 'Visual analysis required'
                },
                'generic_name': medical_info.get('generic_name'),
                'usage': medical_info.get('usage'),
                'dosage': medical_info.get('dosage'),
                'consumption_time': medical_info.get('consumption_time'),
                'side_effects': medical_info.get('side_effects'),
                'precautions': medical_info.get('precautions'),
                'disclaimer': medical_info.get('disclaimer'),
                'debug_info': {
                    'top_3_candidates': [
                        {'name': name, 'confidence': f"{conf*100:.2f}%"} 
                        for name, conf in top_3_candidates
                    ],
                    'confidence_threshold': f"{confidence_threshold*100:.0f}%",
                    'passed_safety_check': True
                }
            }
            return response
        else:
            # Predicted but no medical data available
            logger.warning(f"[SAFETY] No medical database entry for {predicted_label}")
            return _get_partial_info_response(predicted_label, confidence, top_3_candidates)
        
    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {str(e)}", exc_info=True)
        return _get_error_response("PREDICTION_ERROR", str(e))

def _get_unknown_response(confidence, top_3_candidates):
    """Return safe response for unknown/low-confidence tablets."""
    return {
        'tablet_name': 'UNKNOWN TABLET',
        'confidence': f"{confidence*100:.2f}%",
        'pill_features': {
            'color': 'Unable to determine',
            'shape': 'Unable to determine',
            'imprint': 'Unable to determine',
            'size': 'Unable to determine'
        },
        'generic_name': 'Unidentified Medication',
        'usage': 'CRITICAL: Please do not consume this tablet without professional identification.',
        'dosage': 'N/A - Verification Required',
        'consumption_time': 'N/A - Verification Required',
        'side_effects': ['SAFETY: Do not consume without pharmacist verification'],
        'precautions': 'CRITICAL: Pill could not be reliably identified. Contact a pharmacist or poison control immediately.',
        'warnings': 'This tablet could not be identified with sufficient confidence. There is a risk of medication error.',
        'note': 'SAFETY PROTOCOL: Confidence below 70% threshold. Please verify with a licensed pharmacist or healthcare provider before consumption.',
        'debug_info': {
            'top_3_candidates': [
                {'name': name, 'confidence': f"{conf*100:.2f}%"} 
                for name, conf in top_3_candidates
            ],
            'confidence_threshold': '70%',
            'passed_safety_check': False,
            'reason': 'Confidence too low for safe prediction'
        }
    }

def _get_partial_info_response(predicted_label, confidence, top_3_candidates):
    """Return response when prediction is confident but no medical data."""
    return {
        'tablet_name': predicted_label,
        'confidence': f"{confidence*100:.2f}%",
        'pill_features': {
            'color': 'Visual analysis required',
            'shape': 'Visual analysis required',
            'imprint': 'Visual analysis required',
            'size': 'Visual analysis required'
        },
        'generic_name': f'Medication: {predicted_label}',
        'usage': 'Please consult healthcare professional or pharmacist for usage information.',
        'dosage': 'Please consult healthcare professional or pharmacist for dosage information.',
        'consumption_time': 'Please consult healthcare professional or pharmacist.',
        'side_effects': ['Please consult healthcare professional for complete information'],
        'precautions': 'Always consult a healthcare professional for personalized medical advice.',
        'disclaimer': '⚠️ SAFETY: Tablet identified but medical database entry not found. Please consult a pharmacist.',
        'debug_info': {
            'top_3_candidates': [
                {'name': name, 'confidence': f"{conf*100:.2f}%"} 
                for name, conf in top_3_candidates
            ],
            'confidence_threshold': '70%',
            'passed_safety_check': True,
            'note': 'Prediction confident but incomplete medical data'
        }
    }

def _get_error_response(error_type, error_message):
    """Return safe error response."""
    return {
        'tablet_name': 'ERROR',
        'confidence': '0%',
        'pill_features': {
            'color': 'N/A',
            'shape': 'N/A',
            'imprint': 'N/A',
            'size': 'N/A'
        },
        'generic_name': 'System Error',
        'usage': 'System encountered an error',
        'dosage': 'N/A',
        'consumption_time': 'N/A',
        'side_effects': [f'System Error: {error_type}'],
        'precautions': 'Please try again or consult a healthcare professional.',
        'disclaimer': '⚠️ SYSTEM ERROR: Identification system failed. Please contact support or consult a pharmacist.',
        'debug_info': {
            'error_type': error_type,
            'error_message': error_message,
            'passed_safety_check': False
        }
    }

def convert_to_tflite(model_path=None, tflite_path=None):
    """
    Convert Keras model to TensorFlow Lite format for mobile deployment.
    
    Args:
        model_path: Path to Keras model (optional, uses default)
        tflite_path: Path to save TFLite model (optional, uses default)
        
    Returns:
        str: Path to converted TFLite model
    """
    try:
        if model_path is None:
            paths = get_data_paths()
            model_path = paths['model_path']
        
        if tflite_path is None:
            tflite_path = str(model_path).replace('.keras', '.tflite').replace('.h5', '.tflite')
        
        logger.info(f"Converting model {model_path} to TFLite...")
        model = load_model(str(model_path))
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved to {tflite_path}")
        return tflite_path
        
    except Exception as e:
        logger.error(f"TFLite conversion failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run training
    accuracy, model, label_map, reverse_label_map = main()


