import pandas as pd # type: ignore
import os

# Path to CSV (relative to manage.py)
csv_path = os.path.join('frgapp', 'recipes.csv')

# Check if file exists
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("✅ Dataset loaded successfully.")
    print(df.head())
else:
    print("❌ recipes.csv not found at path:", csv_path)
