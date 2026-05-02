"""
Data Validation Utility Module
Validates dataset integrity before training and deployment

Functions:
- validate_images_exist()
- clean_duplicate_test_images()
- sync_csv_with_images()
- generate_validation_report()
"""

import os
import json
import logging
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# ==================== VALIDATION FUNCTIONS ====================

def get_data_paths():
    """Get portable data paths from Django settings or local config."""
    try:
        from django.conf import settings
        if settings.configured:
            BASE_DATA_PATH = Path(settings.BASE_DIR) / 'media' / 'pilldata'
        else:
            BASE_DATA_PATH = Path(__file__).parent.parent.parent / 'media' / 'pilldata'
    except:
        BASE_DATA_PATH = Path(__file__).parent.parent.parent / 'media' / 'pilldata'
    
    return {
        'train_csv': BASE_DATA_PATH / 'Training_set.csv',
        'test_csv': BASE_DATA_PATH / 'Testing_set.csv',
        'train_path': BASE_DATA_PATH / 'train',
        'test_path': BASE_DATA_PATH / 'test',
        'base_path': BASE_DATA_PATH
    }

def check_image_file(image_path, expected_size=(64, 64)):
    """
    Check if image file is valid and has correct dimensions.
    
    Args:
        image_path: Path to image file
        expected_size: Expected (width, height)
        
    Returns:
        dict: {'valid': bool, 'error': str or None, 'size': tuple}
    """
    try:
        if not Path(image_path).exists():
            return {'valid': False, 'error': 'File not found', 'size': None}
        
        img = Image.open(image_path)
        
        if img.format not in ['JPEG', 'PNG', 'JPG']:
            return {'valid': False, 'error': f'Invalid format: {img.format}', 'size': img.size}
        
        if img.size != expected_size:
            return {'valid': False, 'error': f'Wrong size: {img.size} != {expected_size}', 'size': img.size}
        
        # Try reading as array
        np.array(img)
        return {'valid': True, 'error': None, 'size': img.size}
        
    except Exception as e:
        return {'valid': False, 'error': str(e), 'size': None}

def validate_images_exist(csv_path, image_dir):
    """
    Validate all images referenced in CSV exist in image directory.
    
    Args:
        csv_path: Path to CSV file
        image_dir: Path to image directory
        
    Returns:
        dict: Validation results with missing files list
    """
    logger.info(f"Validating images exist in {image_dir}...")
    
    df = pd.read_csv(csv_path)
    image_dir = Path(image_dir)
    
    missing_files = []
    valid_count = 0
    
    for filename in df['filename']:
        file_path = image_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            valid_count += 1
    
    return {
        'total': len(df),
        'valid': valid_count,
        'missing': len(missing_files),
        'missing_files': missing_files,
        'percentage_valid': (valid_count / len(df)) * 100
    }

def check_image_quality(image_dir, sample_size=100):
    """
    Check image quality and consistency in a directory.
    
    Args:
        image_dir: Path to image directory
        sample_size: Number of random images to check
        
    Returns:
        dict: Quality check results
    """
    logger.info(f"Checking image quality in {image_dir}...")
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.JPG')) + \
                  list(image_dir.glob('*.png')) + list(image_dir.glob('*.PNG'))
    
    sample_files = image_files[:sample_size]
    
    sizes = {}
    formats = {}
    corrupted = []
    
    for img_path in sample_files:
        try:
            img = Image.open(img_path)
            size = img.size
            fmt = img.format
            
            sizes[size] = sizes.get(size, 0) + 1
            formats[fmt] = formats.get(fmt, 0) + 1
            
            # Try reading as array to catch corruption
            np.array(img)
            
        except Exception as e:
            corrupted.append((str(img_path), str(e)))
    
    return {
        'total_images': len(image_files),
        'sample_checked': len(sample_files),
        'sizes': sizes,
        'formats': formats,
        'corrupted': corrupted,
        'most_common_size': max(sizes, key=sizes.get) if sizes else None
    }

def find_duplicate_test_images(test_path):
    """
    Find duplicate images in test folder (images with random suffixes).
    
    Args:
        test_path: Path to test image directory
        
    Returns:
        dict: Duplicate groups
    """
    logger.info(f"Finding duplicate images in {test_path}...")
    
    test_path = Path(test_path)
    image_files = list(test_path.glob('*.jpg')) + list(test_path.glob('*.JPG'))
    
    # Group by base filename (without suffix)
    groups = {}
    for img_file in image_files:
        # Extract base name (Image_1003)
        parts = img_file.stem.split('_')
        if len(parts) >= 2:
            base_name = '_'.join(parts[:2])  # e.g., Image_1003
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(img_file.name)
    
    # Find duplicates (groups with more than 1 file)
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    
    return {
        'total_images': len(image_files),
        'duplicate_groups': len(duplicates),
        'duplicate_files': sum(len(v) - 1 for v in duplicates.values()),  # Count extra files
        'duplicates': duplicates
    }

def clean_duplicate_test_images(test_path, keep_original=True):
    """
    Remove duplicate test images (files with random suffixes).
    Keeps the original Image_XXXX.jpg file.
    
    Args:
        test_path: Path to test image directory
        keep_original: If True, keeps original files and removes suffixed ones
        
    Returns:
        dict: Cleaning results
    """
    logger.warning("⚠️  DELETING FILES - This operation cannot be undone!")
    
    test_path = Path(test_path)
    
    # Find duplicates
    dup_info = find_duplicate_test_images(test_path)
    
    deleted_count = 0
    deleted_files = []
    
    for base_name, files in dup_info['duplicates'].items():
        # Sort files - keep the one without extra suffix
        original = None
        duplicates = []
        
        for fname in files:
            # Check if it's the original (Image_XXXX.jpg) or has suffix (Image_XXXX_SUFFIX.jpg)
            if len(fname.split('_')) == 2:  # Original
                original = fname
            else:
                duplicates.append(fname)
        
        # Delete duplicates, keep original
        for dup_file in duplicates:
            file_path = test_path / dup_file
            try:
                file_path.unlink()
                deleted_files.append(dup_file)
                deleted_count += 1
                logger.info(f"Deleted: {dup_file}")
            except Exception as e:
                logger.error(f"Failed to delete {dup_file}: {str(e)}")
    
    return {
        'deleted_count': deleted_count,
        'deleted_files': deleted_files,
        'message': f"Successfully deleted {deleted_count} duplicate test images"
    }

def sync_csv_with_images(csv_path, image_dir, auto_fix=False):
    """
    Synchronize CSV file with actual images in directory.
    
    Args:
        csv_path: Path to CSV file
        image_dir: Path to image directory
        auto_fix: If True, creates new CSV with only existing images
        
    Returns:
        dict: Sync results
    """
    logger.info(f"Synchronizing CSV with images in {image_dir}...")
    
    df = pd.read_csv(csv_path)
    image_dir = Path(image_dir)
    
    existing_files = set()
    missing_files = []
    
    for filename in df['filename']:
        file_path = image_dir / filename
        if file_path.exists():
            existing_files.add(filename)
        else:
            missing_files.append(filename)
    
    valid_df = df[df['filename'].isin(existing_files)].copy()
    
    if auto_fix and len(missing_files) > 0:
        # Save corrected CSV
        backup_path = str(csv_path).replace('.csv', '_backup.csv')
        df.to_csv(backup_path, index=False)
        valid_df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV updated. Backup saved to {backup_path}")
    
    return {
        'total_entries': len(df),
        'existing_entries': len(valid_df),
        'missing_entries': len(missing_files),
        'missing_files': missing_files,
        'percentage_valid': (len(valid_df) / len(df)) * 100 if len(df) > 0 else 0,
        'auto_fixed': auto_fix and len(missing_files) > 0
    }

def generate_validation_report(output_file='validation_report.json'):
    """
    Generate comprehensive dataset validation report.
    
    Args:
        output_file: Path to save JSON report
        
    Returns:
        dict: Complete validation report
    """
    paths = get_data_paths()
    
    logger.info("Generating comprehensive validation report...")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'paths': {
            'train_csv': str(paths['train_csv']),
            'test_csv': str(paths['test_csv']),
            'train_path': str(paths['train_path']),
            'test_path': str(paths['test_path'])
        },
        'training_set': {
            'images_exist': validate_images_exist(paths['train_csv'], paths['train_path']),
            'quality_check': check_image_quality(paths['train_path'])
        },
        'test_set': {
            'images_exist': validate_images_exist(paths['test_csv'], paths['train_path']),
            'quality_check': check_image_quality(paths['train_path']),
            'duplicates': find_duplicate_test_images(paths['test_path'])
        }
    }
    
    # Print report
    print("\n" + "="*70)
    print("DATASET VALIDATION REPORT")
    print("="*70)
    print(json.dumps(report, indent=2))
    print("="*70 + "\n")
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Validation report saved to {output_file}")
    
    return report

# ==================== COMMAND-LINE INTERFACE ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Validation Utility')
    parser.add_argument('--validate', action='store_true', help='Run full validation')
    parser.add_argument('--clean-duplicates', action='store_true', help='Clean duplicate test images')
    parser.add_argument('--sync-csv', action='store_true', help='Sync CSV with existing images')
    parser.add_argument('--report', action='store_true', help='Generate validation report')
    
    args = parser.parse_args()
    
    paths = get_data_paths()
    
    if args.validate:
        print("Training set validation:")
        train_valid = validate_images_exist(paths['train_csv'], paths['train_path'])
        print(json.dumps(train_valid, indent=2))
        
        print("\nTest set validation:")
        test_valid = validate_images_exist(paths['test_csv'], paths['train_path'])
        print(json.dumps(test_valid, indent=2))
    
    if args.clean_duplicates:
        result = clean_duplicate_test_images(paths['test_path'])
        print(json.dumps(result, indent=2))
    
    if args.sync_csv:
        result = sync_csv_with_images(paths['train_csv'], paths['train_path'], auto_fix=True)
        print(json.dumps(result, indent=2))
    
    if args.report:
        generate_validation_report()
    
    if not any([args.validate, args.clean_duplicates, args.sync_csv, args.report]):
        parser.print_help()
