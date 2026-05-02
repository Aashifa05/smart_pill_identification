#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pill CSV Manager - View and Manage Prediction Results
======================================================

This utility allows you to:
1. View all pill predictions stored in CSV
2. Export predictions to different formats
3. Filter predictions by date, pill name, or confidence
4. Generate statistics about predictions
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import csv

class PillCSVManager:
    """Manage pill prediction CSV files."""
    
    def __init__(self, csv_path='media/pilldata/pill_predictions.csv'):
        """
        Initialize CSV manager.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self._load_csv()
    
    def _load_csv(self):
        """Load CSV file into DataFrame."""
        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
                print(f"✓ Loaded {len(self.df)} predictions from {self.csv_path}")
            except Exception as e:
                print(f"✗ Error loading CSV: {str(e)}")
                self.df = pd.DataFrame()
        else:
            print(f"✗ CSV file not found: {self.csv_path}")
            self.df = pd.DataFrame()
    
    def display_all(self, limit=None):
        """
        Display all predictions.
        
        Args:
            limit: Maximum number of rows to display (None = all)
        """
        if self.df.empty:
            print("No predictions available")
            return
        
        display_df = self.df.copy()
        if limit:
            display_df = display_df.tail(limit)
        
        print("\n" + "="*120)
        print("PILL PREDICTION RECORDS")
        print("="*120)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        
        print(display_df.to_string(index=False))
        print("="*120 + "\n")
    
    def filter_by_pill_name(self, pill_name):
        """
        Filter predictions by pill name.
        
        Args:
            pill_name: Name of the pill to filter
        
        Returns:
            pd.DataFrame: Filtered results
        """
        if self.df.empty:
            print("No predictions available")
            return pd.DataFrame()
        
        filtered = self.df[self.df['pill_name'].str.contains(pill_name, case=False, na=False)]
        
        print(f"\n✓ Found {len(filtered)} predictions for '{pill_name}':")
        print(filtered.to_string(index=False))
        return filtered
    
    def filter_by_confidence(self, min_confidence=0.0, max_confidence=100.0):
        """
        Filter predictions by confidence range.
        
        Args:
            min_confidence: Minimum confidence (0-100)
            max_confidence: Maximum confidence (0-100)
        
        Returns:
            pd.DataFrame: Filtered results
        """
        if self.df.empty:
            print("No predictions available")
            return pd.DataFrame()
        
        # Convert confidence strings to numbers
        confidence_values = []
        for conf_str in self.df['confidence']:
            try:
                val = float(str(conf_str).replace('%', ''))
                confidence_values.append(val)
            except:
                confidence_values.append(0)
        
        filtered = self.df[(confidence_values >= min_confidence) & (confidence_values <= max_confidence)]
        
        print(f"\n✓ Found {len(filtered)} predictions with {min_confidence}%-{max_confidence}% confidence:")
        print(filtered.to_string(index=False))
        return filtered
    
    def get_statistics(self):
        """
        Generate statistics about predictions.
        
        Returns:
            dict: Statistics data
        """
        if self.df.empty:
            print("No predictions available")
            return {}
        
        stats = {
            'total_predictions': len(self.df),
            'unique_pills': self.df['pill_name'].nunique(),
            'pills_identified': (self.df['pill_name'] != 'UNKNOWN').sum(),
            'unknown_pills': (self.df['pill_name'] == 'UNKNOWN').sum(),
        }
        
        print("\n" + "="*60)
        print("PREDICTION STATISTICS")
        print("="*60)
        print(f"Total Predictions:      {stats['total_predictions']}")
        print(f"Unique Pills:           {stats['unique_pills']}")
        print(f"Successfully Identified: {stats['pills_identified']}")
        print(f"Unknown Pills:          {stats['unknown_pills']}")
        
        if stats['total_predictions'] > 0:
            success_rate = (stats['pills_identified'] / stats['total_predictions']) * 100
            print(f"Success Rate:           {success_rate:.1f}%")
        
        # Most common pills
        if not self.df.empty:
            pill_counts = self.df['pill_name'].value_counts().head(5)
            print("\nTop 5 Most Identified Pills:")
            for pill, count in pill_counts.items():
                print(f"  - {pill}: {count} times")
        
        print("="*60 + "\n")
        return stats
    
    def export_to_excel(self, output_path='pill_predictions.xlsx'):
        """
        Export predictions to Excel file.
        
        Args:
            output_path: Path to save Excel file
        """
        if self.df.empty:
            print("No predictions to export")
            return False
        
        try:
            self.df.to_excel(output_path, index=False)
            print(f"✓ Exported to {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error exporting to Excel: {str(e)}")
            return False
    
    def export_to_json(self, output_path='pill_predictions.json'):
        """
        Export predictions to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        if self.df.empty:
            print("No predictions to export")
            return False
        
        try:
            self.df.to_json(output_path, orient='records', indent=2)
            print(f"✓ Exported to {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error exporting to JSON: {str(e)}")
            return False
    
    def get_last_n_predictions(self, n=10):
        """
        Get the last N predictions.
        
        Args:
            n: Number of predictions to retrieve
        
        Returns:
            pd.DataFrame: Last N predictions
        """
        if self.df.empty:
            return pd.DataFrame()
        
        return self.df.tail(n)
    
    def clear_csv(self, confirm=True):
        """
        Clear all predictions from CSV.
        
        Args:
            confirm: Require confirmation before clearing
        """
        if confirm:
            response = input(f"Are you sure you want to delete all {len(self.df)} predictions? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return False
        
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp',
                    'pill_name',
                    'confidence',
                    'usage',
                    'dosage',
                    'side_effects',
                    'precautions'
                ])
                writer.writeheader()
            
            self.df = pd.DataFrame()
            print("✓ CSV cleared successfully")
            return True
        except Exception as e:
            print(f"✗ Error clearing CSV: {str(e)}")
            return False


def main():
    """Interactive command-line interface for CSV manager."""
    
    csv_path = 'media/pilldata/pill_predictions.csv'
    manager = PillCSVManager(csv_path)
    
    print("\n" + "="*60)
    print("PILL PREDICTION CSV MANAGER")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Display all predictions")
        print("2. Filter by pill name")
        print("3. Filter by confidence range")
        print("4. View statistics")
        print("5. Export to Excel")
        print("6. Export to JSON")
        print("7. View last 10 predictions")
        print("8. Clear CSV")
        print("9. Exit")
        
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == '1':
            limit = input("Maximum rows to display (press Enter for all): ").strip()
            limit = int(limit) if limit.isdigit() else None
            manager.display_all(limit)
        
        elif choice == '2':
            pill_name = input("Enter pill name to filter: ").strip()
            manager.filter_by_pill_name(pill_name)
        
        elif choice == '3':
            min_conf = input("Minimum confidence (0-100): ").strip()
            max_conf = input("Maximum confidence (0-100): ").strip()
            try:
                min_conf = float(min_conf)
                max_conf = float(max_conf)
                manager.filter_by_confidence(min_conf, max_conf)
            except:
                print("✗ Invalid input")
        
        elif choice == '4':
            manager.get_statistics()
        
        elif choice == '5':
            output = input("Output file path (default: pill_predictions.xlsx): ").strip()
            output = output if output else "pill_predictions.xlsx"
            manager.export_to_excel(output)
        
        elif choice == '6':
            output = input("Output file path (default: pill_predictions.json): ").strip()
            output = output if output else "pill_predictions.json"
            manager.export_to_json(output)
        
        elif choice == '7':
            last_10 = manager.get_last_n_predictions(10)
            if not last_10.empty:
                print("\nLast 10 Predictions:")
                print(last_10.to_string(index=False))
        
        elif choice == '8':
            manager.clear_csv(confirm=True)
            manager._load_csv()  # Reload
        
        elif choice == '9':
            print("Exiting...")
            break
        
        else:
            print("✗ Invalid option")


if __name__ == '__main__':
    main()
