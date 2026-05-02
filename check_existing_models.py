import json
from pathlib import Path

# Check metadata files
for metadata_file in sorted(Path('media/pilldata').glob('model_metadata*.json')):
    with open(metadata_file) as f:
        data = json.load(f)
    print(f'\n{metadata_file.name}:')
    print(f'  Accuracy: {data.get("accuracy", "N/A")}')
    print(f'  Confidence threshold: {data.get("confidence_threshold", "N/A")}')
    print(f'  Classes: {len(data.get("classes", []))}')
    if 'test_results' in data:
        print(f'  Test results: {data["test_results"]}')
