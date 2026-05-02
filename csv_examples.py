#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV Output Quick Start Example
==============================

Shows practical examples of using the pill CSV output feature.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pill_csv_manager import PillCSVManager


def example_1_view_all_predictions():
    """Example 1: View all predictions made so far."""
    print("\n" + "="*70)
    print("EXAMPLE 1: View All Predictions")
    print("="*70)
    
    manager = PillCSVManager('media/pilldata/pill_predictions.csv')
    manager.display_all(limit=5)  # Show last 5


def example_2_filter_by_pill_name():
    """Example 2: Find all predictions for a specific pill."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Filter by Pill Name")
    print("="*70)
    
    manager = PillCSVManager('media/pilldata/pill_predictions.csv')
    
    # Find all Biogesic predictions
    print("\nSearching for 'Biogesic' predictions...")
    biogesic = manager.filter_by_pill_name('Biogesic')
    
    if not biogesic.empty:
        print(f"\nFound {len(biogesic)} Biogesic predictions")
        print(f"Average confidence: {biogesic['confidence'].str.rstrip('%').astype(float).mean():.1f}%")


def example_3_high_confidence_predictions():
    """Example 3: Show only high-confidence predictions."""
    print("\n" + "="*70)
    print("EXAMPLE 3: High Confidence Predictions (> 85%)")
    print("="*70)
    
    manager = PillCSVManager('media/pilldata/pill_predictions.csv')
    high_conf = manager.filter_by_confidence(min_confidence=85, max_confidence=100)
    
    if not high_conf.empty:
        print(f"\nTotal high-confidence predictions: {len(high_conf)}")


def example_4_statistics():
    """Example 4: Get overall statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Prediction Statistics")
    print("="*70)
    
    manager = PillCSVManager('media/pilldata/pill_predictions.csv')
    stats = manager.get_statistics()


def example_5_data_analysis():
    """Example 5: Advanced data analysis using pandas."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Advanced Data Analysis")
    print("="*70)
    
    csv_path = 'media/pilldata/pill_predictions.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        print("Make predictions first to generate CSV data")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("No predictions available")
        return
    
    # Convert confidence to numeric
    df['confidence_num'] = df['confidence'].str.rstrip('%').astype(float)
    
    # Analysis 1: Average confidence
    avg_conf = df['confidence_num'].mean()
    print(f"\nAverage Confidence: {avg_conf:.1f}%")
    
    # Analysis 2: Confidence distribution
    high = len(df[df['confidence_num'] >= 90])
    medium = len(df[(df['confidence_num'] >= 70) & (df['confidence_num'] < 90)])
    low = len(df[df['confidence_num'] < 70])
    
    print(f"\nConfidence Distribution:")
    print(f"  High (≥90%):     {high} predictions")
    print(f"  Medium (70-90%): {medium} predictions")
    print(f"  Low (<70%):      {low} predictions")
    
    # Analysis 3: Unknown pills
    unknown = len(df[df['pill_name'] == 'UNKNOWN'])
    identified = len(df[df['pill_name'] != 'UNKNOWN'])
    
    print(f"\nIdentification Success:")
    print(f"  Identified Pills: {identified}")
    print(f"  Unknown Pills:    {unknown}")
    if len(df) > 0:
        success_rate = (identified / len(df)) * 100
        print(f"  Success Rate:     {success_rate:.1f}%")
    
    # Analysis 4: Pill frequency
    print(f"\nTop Identified Pills:")
    pill_counts = df[df['pill_name'] != 'UNKNOWN']['pill_name'].value_counts().head(5)
    for i, (pill, count) in enumerate(pill_counts.items(), 1):
        print(f"  {i}. {pill}: {count} times")
    
    # Analysis 5: Time-based analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    predictions_by_date = df.groupby('date').size()
    print(f"\nPredictions by Date:")
    for date, count in predictions_by_date.tail(5).items():
        print(f"  {date}: {count} predictions")


def example_6_export_data():
    """Example 6: Export data to different formats."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Export Data")
    print("="*70)
    
    manager = PillCSVManager('media/pilldata/pill_predictions.csv')
    
    # Export to Excel
    print("\nExporting to Excel...")
    excel_path = 'pill_predictions_export.xlsx'
    manager.export_to_excel(excel_path)
    
    # Export to JSON
    print("Exporting to JSON...")
    json_path = 'pill_predictions_export.json'
    manager.export_to_json(json_path)
    
    print(f"\n✓ Files created:")
    print(f"  - {excel_path}")
    print(f"  - {json_path}")


def example_7_recent_predictions():
    """Example 7: Show recent predictions."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Recent Predictions")
    print("="*70)
    
    manager = PillCSVManager('media/pilldata/pill_predictions.csv')
    recent = manager.get_last_n_predictions(10)
    
    if not recent.empty:
        print(f"\nLast {len(recent)} predictions:")
        print(recent.to_string(index=False))
    else:
        print("No predictions available")


def example_8_custom_filtering():
    """Example 8: Custom filtering logic."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Custom Filtering")
    print("="*70)
    
    csv_path = 'media/pilldata/pill_predictions.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("No predictions available")
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Find predictions from today
    today = pd.Timestamp.now().normalize()
    today_predictions = df[df['timestamp'] >= today]
    
    print(f"\nPredictions made today: {len(today_predictions)}")
    
    # Find predictions by specific pills
    specific_pills = df[df['pill_name'].isin(['Biogesic', 'Decolgen', 'Bioflu'])]
    print(f"Predictions for specific pills: {len(specific_pills)}")
    
    # Find low confidence predictions that need review
    df['confidence_num'] = df['confidence'].str.rstrip('%').astype(float)
    low_conf = df[df['confidence_num'] < 60]
    print(f"Low confidence predictions (<60%): {len(low_conf)}")
    
    if not low_conf.empty:
        print("\nLow confidence predictions needing review:")
        print(low_conf[['timestamp', 'pill_name', 'confidence']].to_string(index=False))


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PILL CSV OUTPUT - QUICK START EXAMPLES")
    print("="*70)
    
    examples = [
        ("View All Predictions", example_1_view_all_predictions),
        ("Filter by Pill Name", example_2_filter_by_pill_name),
        ("High Confidence Only", example_3_high_confidence_predictions),
        ("View Statistics", example_4_statistics),
        ("Data Analysis", example_5_data_analysis),
        ("Export Data", example_6_export_data),
        ("Recent Predictions", example_7_recent_predictions),
        ("Custom Filtering", example_8_custom_filtering),
    ]
    
    while True:
        print("\n" + "="*70)
        print("SELECT AN EXAMPLE:")
        print("="*70)
        
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        
        print("9. Run All Examples")
        print("0. Exit")
        
        choice = input("\nSelect option (0-9): ").strip()
        
        try:
            choice = int(choice)
            
            if choice == 0:
                print("Exiting...")
                break
            elif choice == 9:
                for name, func in examples:
                    try:
                        func()
                    except Exception as e:
                        print(f"\n✗ Error in {name}: {str(e)}")
            elif 1 <= choice <= len(examples):
                name, func = examples[choice - 1]
                try:
                    func()
                except Exception as e:
                    print(f"\n✗ Error: {str(e)}")
            else:
                print("✗ Invalid option")
        
        except ValueError:
            print("✗ Invalid input")
    
    print("\n" + "="*70)
    print("For interactive management, run: python pill_csv_manager.py")
    print("="*70)


if __name__ == '__main__':
    main()
