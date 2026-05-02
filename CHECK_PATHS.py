#!/usr/bin/env python
import os
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Detection_and_Analysis_of_Pill.settings')
import django
django.setup()

from Users.utility.requirement import get_data_paths

paths = get_data_paths()
print("\nget_data_paths() results:")
for key, val in paths.items():
    if isinstance(val, Path):
        exists = "✅" if val.exists() else "❌"
    else:
        val = Path(str(val))
        exists = "✅" if val.exists() else "❌"
    print(f"  {key}: {exists} {val}")
