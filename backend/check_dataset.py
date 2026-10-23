
import pandas as pd
from pathlib import Path

# Find the dataset from the project root
file_path = Path("data/raw/url_features_extracted1.csv")

# Check that the file exists
if not file_path.exists():
    print("Dataset not found!")
else:
    # Read the CSV
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())
