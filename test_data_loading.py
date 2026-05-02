#!/usr/bin/env python
"""Quick test to verify data loading works"""

import os
import sys
import re
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

dataset_dir = r"media/pilldata/train"
dataset_path = Path(dataset_dir)

logger.info(f"Checking directory: {dataset_path.absolute()}")
logger.info(f"Directory exists: {dataset_path.exists()}")

if not dataset_path.exists():
    logger.error(f"Directory not found: {dataset_path}")
    sys.exit(1)

# Check for subdirectories
subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
logger.info(f"Found {len(subdirs)} subdirectories")

if subdirs:
    logger.info("NESTED structure detected")
else:
    logger.info("FLAT structure detected")
    
# Find all images
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
all_images = []
for ext in image_extensions:
    all_images.extend(list(dataset_path.glob(ext)))

logger.info(f"Found {len(all_images)} total images")

if all_images:
    logger.info("\nExtracting pill names from filenames...")
    
    def extract_pill_name(filename):
        """Extract pill name from filename with fallback patterns"""
        # Pattern 1: PillName dosage MG (number)
        match = re.match(r'([^0-9]*?)\s+\d+', filename)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: NDC-Code_index or similar
        # Extract part before first underscore or dash
        match = re.match(r'([a-zA-Z0-9]+-?[a-zA-Z0-9]*?)_', filename)
        if match:
            return match.group(1).strip()
        
        # Pattern 3: Just use part before dash
        match = re.match(r'([a-zA-Z0-9]+-[a-zA-Z0-9]+)', filename)
        if match:
            return match.group(1).strip()
        
        return None
    
    pill_names = set()
    
    for img_path in all_images:
        filename = img_path.stem
        pill_name = extract_pill_name(filename)
        if pill_name:
            pill_names.add(pill_name)
        else:
            logger.warning(f"Could not extract pill name from: {filename}")
    
    logger.info(f"\nDetected {len(pill_names)} unique pill classes:")
    for pill_name in sorted(pill_names):
        count = sum(1 for img in all_images if extract_pill_name(img.stem) == pill_name)
        logger.info(f"  {pill_name}: {count} images")
    
    # Try loading a sample image
    sample_img = all_images[0]
    logger.info(f"\nTesting image load on: {sample_img.name}")
    try:
        img = Image.open(sample_img)
        logger.info(f"✓ Successfully loaded image: {img.size} {img.mode}")
    except Exception as e:
        logger.error(f"✗ Failed to load image: {e}")

logger.info("\n✅ Data loading test complete!")
