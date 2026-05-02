import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

data_dir = Path('media/pilldata')
train_csv = data_dir / 'Training_set.csv'

df = pd.read_csv(train_csv)
print('CSV loaded OK')
print('Columns:', df.columns.tolist())

# Test image loading
for idx, row in df.head(10).iterrows():
    img_path = data_dir / 'train' / row['filename']
    try:
        img = Image.open(img_path)
        print(f'{idx}: {row["filename"]} - {img.size} OK')
    except Exception as e:
        print(f'{idx}: {row["filename"]} - ERROR: {e}')
