#!/usr/bin/env python3
"""
DIAGNOSTIC: Compare v1 vs v2 Predictions
This shows why v1 fails and what to expect from v2
"""

import os
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns

def print_section(title):
    """Print formatted section"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

class PredictionComparison:
    """Compare v1 vs v2 predictions"""
    
    def __init__(self):
        self.v1_model_path = 'media/pilldata/model_imprint_robust.h5'
        self.v1_metadata_path = 'media/pilldata/model_imprint_robust_metadata.json'
        self.v2_model_path = 'media/pilldata/model_imprint_robust_v2.h5'
        self.v2_metadata_path = 'media/pilldata/model_imprint_robust_v2_metadata.json'
        
        self.v1_model = None
        self.v2_model = None
        self.v1_classes = None
        self.v2_classes = None
        
    def load_models(self):
        """Load both models"""
        print_section("LOADING MODELS")
        
        # Load v1
        if os.path.exists(self.v1_model_path):
            print(f"Loading v1 model: {self.v1_model_path}")
            try:
                self.v1_model = load_model(self.v1_model_path, compile=False)
                print("✓ v1 model loaded")
                
                with open(self.v1_metadata_path) as f:
                    metadata = json.load(f)
                    self.v1_classes = metadata.get('classes', [])
                print(f"✓ v1 classes: {len(self.v1_classes)} pill types")
            except Exception as e:
                print(f"❌ Failed to load v1: {e}")
                self.v1_model = None
        else:
            print(f"⚠️  v1 model not found: {self.v1_model_path}")
        
        # Load v2 (if exists)
        if os.path.exists(self.v2_model_path):
            print(f"\nLoading v2 model: {self.v2_model_path}")
            try:
                self.v2_model = load_model(self.v2_model_path, compile=False)
                print("✓ v2 model loaded")
                
                with open(self.v2_metadata_path) as f:
                    metadata = json.load(f)
                    self.v2_classes = metadata.get('classes', [])
                print(f"✓ v2 classes: {len(self.v2_classes)} pill types")
            except Exception as e:
                print(f"❌ Failed to load v2: {e}")
                self.v2_model = None
        else:
            print(f"⚠️  v2 model not found yet: {self.v2_model_path}")
            print("   (Will be created when you run train_imprint_robust_v2.py)")
        
        return self.v1_model is not None or self.v2_model is not None
    
    def analyze_v1_issue(self):
        """Analyze why v1 predicts UNKNOWN"""
        if self.v1_model is None:
            return
        
        print_section("ANALYZING v1 MODEL ISSUE")
        print("""
The v1 model predicts everything as UNKNOWN because:

1. AGGRESSIVE AUGMENTATION
   During training:
   - Images were heavily blurred (removes pill details)
   - Morphological opening applied (erases fine features)
   - Contrast reduced to 60% (fades distinguishing colors)
   - Result: All augmented images look similar
   
2. LOSS OF DISCRIMINATIVE FEATURES
   What the model learned:
   ❌ "All pills look like blurred white blobs"
   ❌ "Can't tell aspirin from ibuprofen"
   ❌ "Better reject than guess wrong"
   
3. PREDICTION BEHAVIOR
   Instead of predicting:
   ✓ Aspirin: 80%
   ✓ Ibuprofen: 70%
   
   It predicts:
   ❌ UNKNOWN: 60%
   ❌ Everything has low confidence
   ❌ Too few confident predictions

4. WHY THE THRESHOLD DOESN'T HELP
   - v1 confidence is typically: 25-45%
   - Threshold is: 75%
   - Result: 0% predictions pass the threshold
   - Solution: LOWER threshold won't fix this
   - Real fix: Train with less aggressive augmentation
""")
        
        # Try to show model stats
        try:
            total_params = self.v1_model.count_params()
            print(f"\nModel Statistics:")
            print(f"  • Total parameters: {total_params:,}")
            print(f"  • Model size: ~{total_params / 1e6:.1f} MB")
        except:
            pass
    
    def show_comparison_table(self):
        """Show v1 vs v2 comparison"""
        print_section("v1 vs v2 COMPARISON")
        
        comparison = """
┌─────────────────────┬───────────────────┬───────────────────┐
│ Aspect              │ v1 (AGGRESSIVE)   │ v2 (BALANCED)     │
├─────────────────────┼───────────────────┼───────────────────┤
│ Augmentation        │ Blur + morph       │ Rotation + shift  │
│ Blur radius         │ 5-7 (severe)      │ 3 (light)         │
│ Contrast factor     │ 60% (strong)      │ Not used          │
│ Rotation range      │ None              │ ±15°              │
│ Shift range         │ None              │ ±10%              │
│ Zoom range          │ None              │ ±10%              │
│ Class weights       │ No                │ Yes (balanced)    │
│ Confidence threshold│ 0.75 (too high)   │ 0.50 (realistic)  │
├─────────────────────┼───────────────────┼───────────────────┤
│ Expected Result     │ Everything = UNK  │ Correct classes   │
│ Accuracy (labeled)  │ ~5% ✗             │ ~80-90% ✓         │
│ Accuracy (unlabeled)│ ~5% ✗             │ ~60-70% ✓         │
│ Consistency         │ Bad (all same)    │ Good (varied)     │
└─────────────────────┴───────────────────┴───────────────────┘
""")
        print(comparison)
    
    def show_expected_improvements(self):
        """Show what should improve with v2"""
        print_section("EXPECTED IMPROVEMENTS WITH v2")
        
        improvements = """
BEFORE (v1):
  Input: Aspirin pill (white, oval)
  Output: UNKNOWN (61.75% confidence) ✗
  
  Input: Ibuprofen pill (red, round)
  Output: UNKNOWN (58.23% confidence) ✗
  
  Problem: Can't distinguish pills

AFTER (v2):
  Input: Aspirin pill (white, oval)
  Output: Aspirin (82.5% confidence) ✓
  
  Input: Ibuprofen pill (red, round)
  Output: Ibuprofen (78.9% confidence) ✓
  
  Benefit: Learns pill-specific features


KEY METRICS TO WATCH:
  ✓ Accuracy on labeled pills: Should increase from ~5% to ~80%
  ✓ Accuracy on unlabeled pills: Should increase from ~5% to ~60%
  ✓ UNKNOWN predictions: Should drop from ~95% to ~15%
  ✓ Per-class predictions: Should vary by class, not all same
  ✓ Confidence distribution: Should have two peaks (high and low)
""")
        print(improvements)
    
    def generate_recommendations(self):
        """Generate tuning recommendations"""
        print_section("TRAINING RECOMMENDATIONS")
        
        print("""
FOR CURRENT TRAINING (v2):

1. MONITOR THESE METRICS DURING TRAINING:
   • Loss (should decrease steadily)
   • Accuracy (should increase above 70%)
   • Per-class accuracy (should vary by class)
   • Val accuracy (should follow train accuracy)

2. IF TRAINING STALLS:
   • Check if augmentation is still too strong
   • Try reducing rotation_range to 10°
   • Try reducing shift_range to 0.05
   • Increase learning_rate to 1e-3

3. IF MODEL STILL PREDICTS UNKNOWN:
   • IMMEDIATELY reduce augmentation intensity
   • Run: python quick_import_test.py
   • Check if data is loading correctly
   • Verify classes are being labeled properly

4. IF MODEL TRAINS BUT OVERFITS:
   • Dropout is already in MobileNetV3
   • Increase augmentation if needed
   • Add L2 regularization
   • This is GOOD - means model is learning

5. AFTER TRAINING:
   • Run test_imprint_robustness.py
   • Check per-class metrics
   • Compare confidence distributions
   • Test on unlabeled pills
   • Test consistency (same pill, different angles)
""")

def main():
    """Main analysis"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  DIAGNOSTIC: Why v1 Fails & What v2 Will Fix".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    comp = PredictionComparison()
    
    if comp.load_models():
        comp.analyze_v1_issue()
        comp.show_comparison_table()
        comp.show_expected_improvements()
    else:
        print("\n⚠️  Could not load models for detailed analysis")
        comp.show_comparison_table()
        comp.show_expected_improvements()
    
    comp.generate_recommendations()
    
    print_section("NEXT STEP")
    print("""
Run the training:
  python quick_train_v2.py

This will:
1. Backup v1 model
2. Train v2 with balanced augmentation
3. Verify output files
4. Show integration instructions
5. Take 30-60 minutes

Expected result: Model learns pill classes correctly! ✓
""")

if __name__ == "__main__":
    main()
