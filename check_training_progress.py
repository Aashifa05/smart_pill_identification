#!/usr/bin/env python
"""Monitor training progress"""

import time
import os
from pathlib import Path
import json

def check_training_progress():
    """Check if model exists and training completed"""
    model_path = Path("media/pilldata/model_feature_learning.keras")
    metadata_path = Path("media/pilldata/model_feature_learning_metadata.json")
    
    print("=" * 70)
    print("PILL CLASSIFIER TRAINING STATUS")
    print("=" * 70)
    
    # Check if files exist
    model_exists = model_path.exists()
    metadata_exists = metadata_path.exists()
    
    print(f"\nModel file (.keras): {'✓ EXISTS' if model_exists else '✗ NOT YET'}")
    if model_exists:
        size_mb = model_path.stat().st_size / (1024*1024)
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Modified: {time.ctime(model_path.stat().st_mtime)}")
    
    print(f"Metadata file: {'✓ EXISTS' if metadata_exists else '✗ NOT YET'}")
    if metadata_exists:
        with open(metadata_path) as f:
            meta = json.load(f)
            print(f"  Classes: {len(meta.get('class_names', []))}")
            print(f"  Training samples: {meta.get('training_samples', 'N/A')}")
    
    print("\n" + "=" * 70)
    
    if model_exists and metadata_exists:
        print("✅ TRAINING COMPLETE! Model is ready.")
        print("\nNext steps:")
        print("1. Run diagnostic: python run_model_diagnostic.py --model model_feature_learning.keras")
        print("2. Review results and safety verdict")
        print("3. Deploy to production or retrain if needed")
    else:
        print("⏳ TRAINING IN PROGRESS... (check again in a few minutes)")
        print("\nExpected training time:")
        print("  Data: 1988 images (3x augmentation = 5964 images)")
        print("  Classes: 22 pill types")
        print("  Epochs: 50")
        print("  Estimated time: 40-60 minutes")
    
    print("=" * 70)

if __name__ == '__main__':
    check_training_progress()
