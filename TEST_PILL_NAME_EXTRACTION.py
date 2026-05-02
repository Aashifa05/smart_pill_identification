#!/usr/bin/env python
"""Test the pill name extraction regex from training script"""

import re

def extract_pill_name(filename):
    """Extract pill name from filename - SAME AS training script"""
    # Pattern 1: PillName dosage MG (number)
    match = re.match(r'([^0-9]*?)\s+\d+', filename)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: NDC-Code_index or similar
    match = re.match(r'([a-zA-Z0-9]+-?[a-zA-Z0-9]*?)_', filename)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: Just use part before dash
    match = re.match(r'([a-zA-Z0-9]+-[a-zA-Z0-9]+)', filename)
    if match:
        return match.group(1).strip()
    
    return None

# Test with actual filenames from the dataset
test_files = [
    "Amoxicillin 500 MG (1).jpg",
    "apixaban 2.5 MG (1).jpg",
    "aprepitant 80 MG (1).jpg",
    "benzonatate 100 MG (1).jpg",
    "Calcitriol 0.00025 MG (1).jpg",
    "0002-3228_0_1.jpg",
    "69387-119_0_0.jpg",
]

print("\n" + "=" * 100)
print("PILL NAME EXTRACTION TEST")
print("=" * 100)

print("\n🔍 Testing filename parsing:")
for filename in test_files:
    # Remove extension for testing
    name_only = filename.rsplit('.', 1)[0]
    extracted = extract_pill_name(name_only)
    print(f"\n  Input:     {filename}")
    print(f"  Extracted: {extracted}")
    
    # Check if it's the right format
    expected_start = filename.split('(')[0].strip() if '(' in filename else filename.split('_')[0]
    
    if extracted and expected_start in extracted:
        print(f"  ✅ Looks correct")
    else:
        print(f"  ❌ WRONG! Expected something like: {expected_start}")

print("\n" + "=" * 100)
print("\n⚠️  ISSUE FOUND:")
print("\n  Pattern 1 in extract_pill_name uses: r'([^0-9]*?)\\s+\\d+'")
print("  This captures text BEFORE the space+number")
print("  For 'Amoxicillin 500 MG (1)' it only extracts: 'Amoxicillin'")
print("  But CSV expects: 'Amoxicillin 500 MG'")
print("\n  This causes:")
print("  ❌ Wrong class names extracted from filenames")
print("  ❌ Label map mismatch with actual pill names")
print("  ❌ All pills might be collapsed into fewer classes")
print("\n" + "=" * 100 + "\n")
