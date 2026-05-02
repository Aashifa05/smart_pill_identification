import os
import sys
import json
from pathlib import Path
import numpy as np

# Test data loading
print("=" * 60)
print("TEST: Model Training Verification")
print("=" * 60)

# Check if trained model exists
model_path = Path("c:/Users/THINKBOOK/OneDrive/Desktop/Admin/media/pilldata/model_imprint_robust_v2.h5")
metadata_path = Path("c:/Users/THINKBOOK/OneDrive/Desktop/Admin/media/pilldata/model_imprint_robust_v2_metadata.json")

print(f"\n1. Checking model file: {model_path}")
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024**2)
    print(f"   ✓ Found! Size: {size_mb:.1f} MB")
else:
    print(f"   ✗ Not found")

print(f"\n2. Checking metadata file: {metadata_path}")
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"   ✓ Found! Classes: {len(metadata.get('classes', []))}")
    print(f"     Confidence threshold: {metadata.get('confidence_threshold', 'N/A')}")
else:
    print(f"   ✗ Not found")

# Check data paths
print(f"\n3. Checking data paths:")
train_csv = Path("c:/Users/THINKBOOK/OneDrive/Desktop/Admin/media/pilldata/Training_set.csv")
test_csv = Path("c:/Users/THINKBOOK/OneDrive/Desktop/Admin/media/pilldata/Testing_set.csv")
train_path = Path("c:/Users/THINKBOOK/OneDrive/Desktop/Admin/media/pilldata/train")

print(f"   Training CSV: {'✓' if train_csv.exists() else '✗'}")
print(f"   Testing CSV: {'✓' if test_csv.exists() else '✗'}")
print(f"   Images folder: {'✓' if train_path.exists() else '✗'}")

if train_path.exists():
    img_count = len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.JPG")))
    print(f"   Image count: {img_count}")

print(f"\n4. Training status:")
print(f"   Model trained: {'✓ YES' if model_path.exists() else '✗ NO'}")
print(f"   Metadata saved: {'✓ YES' if metadata_path.exists() else '✗ NO'}")
print(f"   Ready to use: {'✓ YES' if (model_path.exists() and metadata_path.exists()) else '✗ NO'}")

print("\n" + "=" * 60)
