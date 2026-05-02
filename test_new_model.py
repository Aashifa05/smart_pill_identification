"""Test the new trained model (89.05% accuracy)"""

from Users.utility.requirement import predictions
from pathlib import Path

# Get a test image from the test folder
test_path = Path('media/pilldata/test')
test_images = []
for drug_folder in test_path.iterdir():
    if drug_folder.is_dir():
        images = list(drug_folder.glob('*.jpg'))
        if images:
            test_images.append((drug_folder.name, str(images[0])))
            if len(test_images) >= 5:
                break

print('\n=== TESTING NEW MODEL (89.05% ACCURACY) ===\n')

correct = 0
total = 0
for drug_name, image_path in test_images:
    try:
        result = predictions(image_path)
        pill_name = result['pill_name']
        confidence = result['confidence']
        is_correct = pill_name.lower() == drug_name.lower()
        correct += int(is_correct)
        total += 1
        
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f'{status}')
        print(f'  Expected: {drug_name}')
        print(f'  Predicted: {pill_name}')
        print(f'  Confidence: {confidence}')
        print('-' * 50)
    except Exception as e:
        print(f'Error: {e}')
        print('-' * 50)

print(f'\nResults: {correct}/{total} correct')
