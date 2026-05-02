#!/usr/bin/env python
"""
Quick MobileNetV3 Model Retraining Script
This script retrains the pill detection model with the new MobileNetV3 architecture.
"""
import os
import sys
import shutil
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.requirement import main

def backup_old_model():
    """Backup the old model files"""
    media_path = Path('media/pilldata')
    
    # Backup old model.h5
    old_model = media_path / 'model.h5'
    if old_model.exists():
        backup_path = media_path / 'model_old_cnn_backup.h5'
        shutil.copy(old_model, backup_path)
        print(f"✓ Old model backed up to: {backup_path}")
    
    # Backup old metadata
    old_metadata = media_path / 'model_metadata.json'
    if old_metadata.exists():
        backup_metadata = media_path / 'model_metadata_old_backup.json'
        shutil.copy(old_metadata, backup_metadata)
        print(f"✓ Old metadata backed up to: {backup_metadata}")

if __name__ == '__main__':
    print("=" * 70)
    print("PILL DETECTION MODEL - MobileNetV3 RETRAINING")
    print("=" * 70)
    
    # Backup old models
    print("\nBacking up old model files...")
    backup_old_model()
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING WITH MobileNetV3Large + Transfer Learning")
    print("=" * 70)
    print("\nNote: First run will download ImageNet weights (~5MB)")
    print("Training will be faster due to pre-trained features\n")
    
    try:
        # Run training
        accuracy, model, label_map, reverse_label_map = main()
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Final Test Accuracy: {accuracy*100:.2f}%")
        print(f"Model saved to: media/pilldata/model.h5")
        print(f"Metadata saved to: media/pilldata/model_metadata.json")
        print("\nYour app is now ready with MobileNetV3!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TRAINING FAILED: {str(e)}")
        print("Check the error messages above for details")
        sys.exit(1)
