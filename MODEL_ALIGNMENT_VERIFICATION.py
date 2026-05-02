#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODEL ALIGNMENT VERIFICATION
Verify that training, evaluation, and deployment use the SAME model file
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model

print("\n" + "=" * 100)
print("MODEL ALIGNMENT VERIFICATION - ENSURING TRAINING, EVALUATION, AND DEPLOYMENT USE SAME MODEL")
print("=" * 100)

# ============================================================================
# PART 1: IDENTIFY ALL MODEL PATHS
# ============================================================================
print("\n" + "-" * 100)
print("PART 1: IDENTIFY ALL MODEL PATHS USED IN CODEBASE")
print("-" * 100)

# Get current working directory
project_root = Path(__file__).parent.parent  # pill_project/
admin_dir = project_root / 'Admin'

print(f"\nProject Root: {project_root}")
print(f"Admin Directory: {admin_dir}")

# Define expected paths
paths_dict = {
    "Training Output (train_feature_learning.py line 270)": 
        project_root / 'media' / 'pilldata' / 'model_feature_learning.keras',
    
    "Training Metadata": 
        project_root / 'media' / 'pilldata' / 'model_feature_learning_metadata.json',
    
    "Deployment Model (Users/utility/requirement.py line 322)": 
        project_root / 'media' / 'pilldata' / 'model_feature_learning_final_best.keras',
    
    "Deployment Metadata": 
        project_root / 'media' / 'pilldata' / 'model_feature_learning_final_metadata.json',
    
    "Visualization Script Uses (regenerate_visualizations.py)": 
        project_root / 'media' / 'pilldata' / 'model_feature_learning_final_best.keras',
}

print("\n📍 EXPECTED MODEL PATHS:")
print("-" * 100)

for description, path in paths_dict.items():
    exists = "✅ EXISTS" if path.exists() else "❌ MISSING"
    size = f"{path.stat().st_size / (1024*1024):.2f} MB" if path.exists() else "N/A"
    print(f"\n{description}")
    print(f"  Path: {path}")
    print(f"  Status: {exists}")
    if path.exists():
        print(f"  Size: {size}")

# ============================================================================
# PART 2: COMPUTE FILE HASHES
# ============================================================================
print("\n" + "-" * 100)
print("PART 2: COMPUTE SHA256 HASHES TO CONFIRM IDENTICAL FILES")
print("-" * 100)

def compute_sha256(filepath):
    """Compute SHA256 hash of file"""
    if not filepath.exists():
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

model_paths_to_check = [
    ("Training Output", project_root / 'media' / 'pilldata' / 'model_feature_learning.keras'),
    ("Deployment/Visualization", project_root / 'media' / 'pilldata' / 'model_feature_learning_final_best.keras'),
]

hashes = {}
print("\n🔐 SHA256 HASHES:")
print("-" * 100)

for name, path in model_paths_to_check:
    hash_val = compute_sha256(path)
    hashes[name] = hash_val
    if hash_val:
        print(f"\n{name}:")
        print(f"  Path: {path}")
        print(f"  SHA256: {hash_val}")
    else:
        print(f"\n{name}:")
        print(f"  Path: {path}")
        print(f"  ❌ FILE NOT FOUND")

# Check if hashes match
training_hash = hashes.get("Training Output")
deployment_hash = hashes.get("Deployment/Visualization")

print("\n" + "=" * 100)
if training_hash and deployment_hash:
    if training_hash == deployment_hash:
        print("✅ HASHES MATCH - Files are identical")
    else:
        print("❌ HASHES DO NOT MATCH - Files are DIFFERENT!")
        print(f"   Training file hash:    {training_hash}")
        print(f"   Deployment file hash:  {deployment_hash}")
elif not training_hash:
    print("⚠️  Training output file does not exist")
elif not deployment_hash:
    print("⚠️  Deployment file does not exist")
print("=" * 100)

# ============================================================================
# PART 3: LOAD MODELS AND CHECK ARCHITECTURE
# ============================================================================
print("\n" + "-" * 100)
print("PART 3: LOAD MODELS AND VERIFY ARCHITECTURE")
print("-" * 100)

models_loaded = {}

for name, path in model_paths_to_check:
    if path.exists():
        try:
            print(f"\nLoading {name} model...")
            model = load_model(str(path))
            models_loaded[name] = model
            
            print(f"  ✅ Model loaded successfully")
            print(f"     Input Shape: {model.input_shape}")
            print(f"     Output Shape: {model.output_shape}")
            print(f"     Total Parameters: {model.count_params():,}")
            print(f"     Model Type: {type(model).__name__}")
            
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            models_loaded[name] = None
    else:
        print(f"\n{name}: ❌ Model file not found at {path}")
        models_loaded[name] = None

# ============================================================================
# PART 4: EXTRACT TRAINING HISTORY FROM METADATA
# ============================================================================
print("\n" + "-" * 100)
print("PART 4: EXTRACT TRAINING INFORMATION FROM METADATA")
print("-" * 100)

metadata_paths = [
    ("Training Metadata", project_root / 'media' / 'pilldata' / 'model_feature_learning_metadata.json'),
    ("Deployment Metadata", project_root / 'media' / 'pilldata' / 'model_feature_learning_final_metadata.json'),
]

for name, meta_path in metadata_paths:
    print(f"\n{name}:")
    print(f"  Path: {meta_path}")
    
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"  ✅ Metadata loaded")
            print(f"  Contents:")
            for key, value in metadata.items():
                if isinstance(value, dict) and len(str(value)) > 100:
                    print(f"    - {key}: {{...}} ({len(value)} items)")
                elif isinstance(value, (list, dict)) and len(str(value)) > 50:
                    print(f"    - {key}: {str(value)[:80]}...")
                else:
                    print(f"    - {key}: {value}")
        except Exception as e:
            print(f"  ❌ Error loading metadata: {e}")
    else:
        print(f"  ❌ Metadata file not found")

# ============================================================================
# PART 5: TRACE WHERE MODELS ARE LOADED IN CODE
# ============================================================================
print("\n" + "-" * 100)
print("PART 5: CODE LOCATIONS - WHERE MODELS ARE LOADED")
print("-" * 100)

code_locations = {
    "🔵 TRAINING": {
        "File": "pill_project/Admin/train_feature_learning.py",
        "Line": 270,
        "Code": 'model_output_path=\'media/pilldata/model_feature_learning.keras\'',
        "Uses": "model_feature_learning.keras"
    },
    "🟡 VISUALIZATION (regenerate_visualizations.py)": {
        "File": "pill_project/Admin/regenerate_visualizations.py",
        "Line": 40,
        "Code": 'paths = get_data_paths()',
        "Uses": "get_data_paths() → model_feature_learning_final_best.keras"
    },
    "🟢 DEPLOYMENT (Users/utility/requirement.py)": {
        "File": "pill_project/Admin/Users/utility/requirement.py",
        "Line": 322,
        "Code": "'model_path': BASE_DATA_PATH / 'model_feature_learning_final_best.keras'",
        "Uses": "model_feature_learning_final_best.keras"
    },
    "🔴 CONFUSION MATRIX": {
        "File": "pill_project/Admin/train_feature_learning.py",
        "Line": 533,
        "Code": "model.save(self.model_output_path)",
        "Uses": "Saves to model_feature_learning.keras (from line 270)"
    }
}

for location, details in code_locations.items():
    print(f"\n{location}")
    for key, value in details.items():
        print(f"  {key}: {value}")

# ============================================================================
# PART 6: FINAL ASSESSMENT
# ============================================================================
print("\n" + "=" * 100)
print("FINAL ASSESSMENT")
print("=" * 100)

training_path = project_root / 'media' / 'pilldata' / 'model_feature_learning.keras'
deployment_path = project_root / 'media' / 'pilldata' / 'model_feature_learning_final_best.keras'

training_exists = training_path.exists()
deployment_exists = deployment_path.exists()

print(f"\n📦 MODEL FILE STATUS:")
print(f"  Training saves to: {'✅' if training_exists else '❌'} {training_path.name}")
print(f"  Deployment loads from: {'✅' if deployment_exists else '❌'} {deployment_path.name}")
print(f"  Visualization loads from: {'✅' if deployment_exists else '❌'} {deployment_path.name}")

print(f"\n🔗 ALIGNMENT STATUS:")
if training_path == deployment_path:
    print("  ✅ All paths are IDENTICAL - No issues")
else:
    print("  ⚠️  CRITICAL: Training and Deployment use DIFFERENT files!")
    print(f"     Training:    {training_path.name}")
    print(f"     Deployment:  {deployment_path.name}")
    
    if training_exists and deployment_exists:
        if training_hash == deployment_hash:
            print("     ✅ BUT: Files are identical (same content)")
        else:
            print("     ❌ AND: Files have DIFFERENT content (different models!)")
    elif training_exists and not deployment_exists:
        print("     ❌ Deployment file missing - using wrong path!")
    elif not training_exists and deployment_exists:
        print("     ⚠️  Training file not found - only deployment file exists")

print("\n" + "=" * 100)
if training_hash == deployment_hash and training_hash is not None:
    print("✅ CONCLUSION: Same model file used everywhere (files are identical)")
elif training_hash != deployment_hash and training_hash is not None and deployment_hash is not None:
    print("❌ CONCLUSION: DIFFERENT model files - training and deployment use different models!")
elif not training_exists or not deployment_exists:
    print("⚠️  CONCLUSION: One or both model files are missing")
print("=" * 100 + "\n")

# ============================================================================
# PART 7: IDENTIFY EPOCH COUNT FROM TRAINING
# ============================================================================
print("\n" + "-" * 100)
print("PART 7: TRAINING EPOCH COUNT FROM HISTORY")
print("-" * 100)

# Try to load training history from checkpoint or metadata
history_file = project_root / 'media' / 'pilldata' / 'training_history.json'
if history_file.exists():
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        epochs_completed = len(history.get('accuracy', []))
        print(f"\n✅ Training history found: {history_file}")
        print(f"   Epochs completed: {epochs_completed}")
        if 'accuracy' in history:
            print(f"   Final training accuracy: {history['accuracy'][-1]:.4f}")
        if 'val_accuracy' in history:
            print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    except Exception as e:
        print(f"❌ Error loading history: {e}")
else:
    print(f"⚠️  Training history file not found: {history_file}")
    print(f"   Checking metadata for epoch information...")
    
    for name, meta_path in metadata_paths:
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    if 'epochs_trained' in metadata:
                        print(f"   {meta_path.name}: {metadata['epochs_trained']} epochs")
            except:
                pass

print("\n" + "=" * 100)
