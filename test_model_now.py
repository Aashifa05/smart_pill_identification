#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TEST_MODEL_NOW.py
=================

Simple script to quickly test your model right now.
Just run: python test_model_now.py

This will:
1. Find your trained models
2. Let you pick which one to test
3. Run the diagnostic
4. Show you the results immediately
"""

import os
import sys
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              PILL CLASSIFIER MODEL QUICK EVALUATION                        ║
║                     Test Your Model in 5 Minutes                           ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# Check if models exist
models_dir = "media/pilldata"
if not os.path.exists(models_dir):
    print(f"❌ ERROR: {models_dir} not found")
    sys.exit(1)

# Find all keras models
models = [f for f in os.listdir(models_dir) if f.endswith('.keras')]

if not models:
    print(f"❌ ERROR: No .keras models found in {models_dir}")
    sys.exit(1)

print(f"✓ Found {len(models)} trained models:\n")

for idx, model in enumerate(sorted(models), 1):
    size_mb = os.path.getsize(os.path.join(models_dir, model)) / (1024*1024)
    print(f"  {idx}. {model:<30} ({size_mb:.1f} MB)")

print(f"\n{'='*70}")
print("WHICH MODEL DO YOU WANT TO TEST?\n")

# Get user choice
while True:
    try:
        choice = input(f"Enter model number (1-{len(models)}) or model name [default: 1]: ").strip()
        
        if not choice:
            choice = "1"
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model = sorted(models)[idx]
                break
            else:
                print(f"❌ Invalid number. Please enter 1-{len(models)}")
        else:
            if choice in models:
                selected_model = choice
                break
            else:
                print(f"❌ Model '{choice}' not found. Try again.")
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'='*70}")
print(f"✓ Selected model: {selected_model}")
print(f"✓ Starting diagnostic...\n")
print(f"{'='*70}\n")

# Run diagnostic
os.system(f"python run_model_diagnostic.py --model {selected_model}")
