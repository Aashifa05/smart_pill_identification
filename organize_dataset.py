#!/usr/bin/env python
"""
Organize Kaggle 1K Pill Dataset and create CSV files for training
"""
import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_pill_dataset():
    """
    Organize the 1K Pill Image Dataset into proper folder structure
    and create training/testing CSV files
    """
    # Paths
    source_dir = Path(r"C:\Users\ezhil\Desktop\Detection_and_Analysis_of_Pill\media\1K-Pill-Image-DataSet")
    train_dir = Path(r"C:\Users\ezhil\Desktop\Detection_and_Analysis_of_Pill\media\pilldata\train")
    test_dir = Path(r"C:\Users\ezhil\Desktop\Detection_and_Analysis_of_Pill\media\pilldata\test")

    print("🔍 Analyzing dataset structure...")

    # Get all image files
    image_files = []
    for file in source_dir.glob("*.jpg"):
        if file.name.startswith("(") and ")" in file.name:
            # Extract class number from filename like "(0)r15.jpg" -> class 0
            try:
                class_num = int(file.name.split(")")[0][1:])  # Extract number between ( and )
                image_files.append((file, class_num))
            except:
                continue

    print(f"📊 Found {len(image_files)} images across {len(set(c for _, c in image_files))} classes")

    # Group by class
    class_groups = {}
    for file_path, class_num in image_files:
        if class_num not in class_groups:
            class_groups[class_num] = []
        class_groups[class_num].append(file_path)

    # Create train/test split for each class
    train_data = []
    test_data = []

    for class_num, files in class_groups.items():
        if len(files) < 2:
            # If only 1 image, put it in training
            train_files = files
            test_files = []
        else:
            # Split 80/20
            train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

        # Add to CSV data
        for file_path in train_files:
            train_data.append({
                'filename': file_path.name,
                'label': class_num
            })

        for file_path in test_files:
            test_data.append({
                'filename': file_path.name,
                'label': class_num
            })

        # Copy files to appropriate directories
        class_train_dir = train_dir / str(class_num)
        class_test_dir = test_dir / str(class_num)
        class_train_dir.mkdir(exist_ok=True)
        class_test_dir.mkdir(exist_ok=True)

        for file_path in train_files:
            shutil.copy2(file_path, class_train_dir / file_path.name)

        for file_path in test_files:
            shutil.copy2(file_path, class_test_dir / file_path.name)

    # Create CSV files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_csv_path = Path(r"C:\Users\ezhil\Desktop\Detection_and_Analysis_of_Pill\media\pilldata\Training_set.csv")
    test_csv_path = Path(r"C:\Users\ezhil\Desktop\Detection_and_Analysis_of_Pill\media\pilldata\Testing_set.csv")

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print("✅ Dataset organization complete!")
    print(f"📁 Training images: {len(train_df)} (in {train_dir})")
    print(f"📁 Testing images: {len(test_df)} (in {test_dir})")
    print(f"🏷️  Classes: {len(class_groups)}")
    print(f"💾 Training CSV: {train_csv_path}")
    print(f"💾 Testing CSV: {test_csv_path}")

    return len(class_groups)

if __name__ == "__main__":
    num_classes = organize_pill_dataset()
    print(f"\\n🎯 Ready to create model with {num_classes} classes!")