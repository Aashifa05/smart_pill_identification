#!/usr/bin/env python
"""
Process new pharmaceutical dataset with class-based folder structure
Converts folder structure to CSV format compatible with the project
"""
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_new_dataset():
    """
    Process the new dataset with folder structure:
    media/train/<drug_name>/ - contains drug images
    media/valid/<drug_name>/ - contains validation images
    
    Creates:
    - media/pilldata/train/ - organized train images
    - media/pilldata/test/ - organized test images
    - media/pilldata/Training_set.csv
    - media/pilldata/Testing_set.csv
    """
    
    base_path = Path(r"C:\Users\ezhil\Desktop\Detection_and_Analysis_of_Pill\media")
    source_train_dir = base_path / "train"
    source_valid_dir = base_path / "valid"
    
    target_train_dir = base_path / "pilldata" / "train"
    target_test_dir = base_path / "pilldata" / "test"
    output_csv_dir = base_path / "pilldata"
    
    # Create target directories
    target_train_dir.mkdir(parents=True, exist_ok=True)
    target_test_dir.mkdir(parents=True, exist_ok=True)
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🔍 Scanning dataset structure...")
    
    # Collect all images with their class labels
    all_data = []
    class_mapping = {}
    class_id = 0
    
    # Process training classes
    for class_folder in sorted(source_train_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        if class_name not in class_mapping:
            class_mapping[class_name] = class_id
            class_id += 1
        
        class_label = class_mapping[class_name]
        
        # Get all images in this class
        for image_file in class_folder.glob("*.jpg"):
            all_data.append({
                'filename': image_file.name,
                'label': class_label,
                'class_name': class_name,
                'source': image_file,
                'split': 'train'
            })
    
    # Process validation classes
    for class_folder in sorted(source_valid_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        if class_name not in class_mapping:
            class_mapping[class_name] = class_id
            class_id += 1
        
        class_label = class_mapping[class_name]
        
        # Get all images in this class
        for image_file in class_folder.glob("*.jpg"):
            all_data.append({
                'filename': image_file.name,
                'label': class_label,
                'class_name': class_name,
                'source': image_file,
                'split': 'valid'
            })
    
    logger.info(f"📊 Found {len(all_data)} images across {len(class_mapping)} classes")
    logger.info(f"\nClass Mapping:")
    for name, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
        logger.info(f"  {idx}: {name}")
    
    # Create dataframes for train and test sets
    train_records = [d for d in all_data if d['split'] == 'train']
    test_records = [d for d in all_data if d['split'] == 'valid']
    
    logger.info(f"\n📈 Dataset Split:")
    logger.info(f"  Training samples: {len(train_records)}")
    logger.info(f"  Test/Validation samples: {len(test_records)}")
    
    # Copy images to destination folders
    logger.info("\n📁 Copying training images...")
    train_data_for_csv = []
    for record in train_records:
        source_file = record['source']
        dest_file = target_train_dir / record['filename']
        
        # Avoid copying to same location
        if source_file != dest_file:
            shutil.copy2(source_file, dest_file)
        
        train_data_for_csv.append({
            'filename': record['filename'],
            'label': record['class_name']
        })
    
    logger.info(f"✅ Copied {len(train_data_for_csv)} training images")
    
    logger.info("\n📁 Copying test images...")
    test_data_for_csv = []
    for record in test_records:
        source_file = record['source']
        dest_file = target_test_dir / record['filename']
        
        # Avoid copying to same location
        if source_file != dest_file:
            shutil.copy2(source_file, dest_file)
        
        test_data_for_csv.append({
            'filename': record['filename'],
            'label': record['class_name']
        })
    
    logger.info(f"✅ Copied {len(test_data_for_csv)} test images")
    
    # Create CSV files
    train_df = pd.DataFrame(train_data_for_csv)
    test_df = pd.DataFrame(test_data_for_csv)
    
    train_csv_path = output_csv_dir / "Training_set.csv"
    test_csv_path = output_csv_dir / "Testing_set.csv"
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    logger.info(f"\n✅ Created training CSV: {train_csv_path}")
    logger.info(f"✅ Created test CSV: {test_csv_path}")
    
    # Save class mapping
    mapping_df = pd.DataFrame([
        {'class_id': idx, 'class_name': name} 
        for name, idx in sorted(class_mapping.items(), key=lambda x: x[1])
    ])
    
    mapping_csv_path = output_csv_dir / "class_mapping.csv"
    mapping_df.to_csv(mapping_csv_path, index=False)
    logger.info(f"✅ Created class mapping: {mapping_csv_path}")
    
    # Dataset statistics
    logger.info("\n📊 Dataset Statistics:")
    logger.info(f"  Total images: {len(train_data_for_csv) + len(test_data_for_csv)}")
    logger.info(f"  Number of classes: {len(class_mapping)}")
    logger.info(f"  Train/Test ratio: {len(train_data_for_csv)}/{len(test_data_for_csv)}")
    
    # Class distribution
    logger.info("\n📈 Class Distribution (Training Set):")
    train_class_dist = train_df['label'].value_counts().sort_index()
    for class_name, count in train_class_dist.items():
        logger.info(f"  {class_name}: {count} images")
    
    return True

if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Processing New Pharmaceutical Dataset")
        logger.info("=" * 60)
        process_new_dataset()
        logger.info("\n" + "=" * 60)
        logger.info("✅ Dataset processing completed successfully!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"❌ Error processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
