#!/usr/bin/env python3
"""
QUICK START: Train v2 Balanced Model
This script trains the model using balanced augmentation approach.
Expected time: 30-60 minutes
Expected result: Model learns pill classes correctly
"""

import os
import sys
import json
from pathlib import Path

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_model_exists():
    """Check if previous model exists"""
    old_model = 'media/pilldata/model_imprint_robust.h5'
    if os.path.exists(old_model):
        print_section("OLD MODEL DETECTED")
        print(f"Found: {old_model}")
        print("⚠️  This is the AGGRESSIVE v1 model that predicts everything as UNKNOWN")
        response = input("\nBackup and delete old model? (y/n): ").strip().lower()
        if response == 'y':
            backup_name = old_model.replace('.h5', '_v1_backup.h5')
            import shutil
            shutil.copy(old_model, backup_name)
            os.remove(old_model)
            print(f"✓ Backed up to: {backup_name}")
            print(f"✓ Deleted: {old_model}")
        else:
            print("⚠️  Keeping old model. New model will be saved as model_imprint_robust_v2.h5")

def start_training():
    """Start the balanced training"""
    print_section("STARTING v2 BALANCED TRAINING")
    print("\nConfiguration:")
    print("  • Augmentation: Moderate (rotation 15°, shift 10%, zoom 10%)")
    print("  • Class weights: Balanced (for imbalanced data)")
    print("  • Confidence threshold: 0.50 (realistic)")
    print("  • Learning rate: 5e-4")
    print("  • Early stopping: 15 epochs patience")
    print("  • Max epochs: 60")
    print("\nTraining started. This will take 30-60 minutes...")
    print("Monitor the loss and accuracy values below:\n")
    
    # Import and run the training script
    try:
        from train_imprint_robust_v2 import main
        main()
        return True
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_output():
    """Verify training output"""
    print_section("VERIFYING OUTPUT")
    
    required_files = [
        'media/pilldata/model_imprint_robust_v2.h5',
        'media/pilldata/model_imprint_robust_v2_metadata.json'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)
            print(f"✓ {file_path} ({size:.1f} MB)")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✅ All output files created successfully!")
        return True
    else:
        print("\n❌ Some output files are missing")
        return False

def show_next_steps():
    """Show next steps"""
    print_section("NEXT STEPS")
    print("""
1. TEST THE MODEL
   python test_imprint_robustness.py
   
   This will:
   - Show per-class accuracy
   - Compare v1 vs v2 results
   - Display confidence distributions
   - Show improvement percentage

2. INTEGRATE INTO DJANGO VIEWS
   Update your views.py to use:
   
   from Users.utility.improved_imprint_aware_predictor import ImprovedImprintAwarePredictor
   
   predictor = ImprovedImprintAwarePredictor(
       'media/pilldata/model_imprint_robust_v2.h5',
       'media/pilldata/model_imprint_robust_v2_metadata.json'
   )
   
   result = predictor.predict(image_path, confidence_threshold=0.50)

3. TEST ON REAL IMAGES
   Try predictions on:
   - Pills WITH imprints (should be 70-90% confident)
   - Pills WITHOUT imprints (should be 50-70% confident)
   - Different pill types (should get different predictions)
   - Unknown pills (should be <30% confident)

4. MONITOR PREDICTIONS
   Watch for:
   ✓ Pills correctly identified
   ✓ Similar pills get consistent labels
   ✓ Confidence scores in reasonable range
   ✓ UNKNOWN count much lower than v1
""")

def main():
    """Main entry point"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PILL IDENTIFICATION - v2 BALANCED TRAINING QUICKSTART".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("""
This script will train the model using balanced augmentation that:
  ✓ Keeps pill shape/color information
  ✓ Adds realistic variations (not aggressive removal)
  ✓ Uses class weighting for imbalance
  ✓ Should fix the "everything is UNKNOWN" problem
""")
    
    # Check old model
    check_model_exists()
    
    # Train
    print("\n")
    response = input("Ready to start training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    success = start_training()
    
    if success:
        # Verify output
        if verify_output():
            # Show next steps
            show_next_steps()
            print_section("TRAINING COMPLETE")
            print("✅ Model trained successfully!")
            print("✅ Ready for testing and integration")
    else:
        print_section("TRAINING FAILED")
        print("❌ Check error messages above")

if __name__ == "__main__":
    main()
