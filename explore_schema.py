import pandas as pd
import os

sample_file = "/Users/arya_vachhani/Downloads/Reddit Data/raw data copy/2019/JAN/anijan19.csv"
print(f"Loading {sample_file}...")

try:
    df = pd.read_csv(sample_file)
    print("Columns:")
    print(df.columns.tolist())
    print("\nSample Data:")
    print(df.head(2))
    print("\nInfo:")
    print(df.info())
except Exception as e:
    print(f"Error reading with default settings: {e}")
    # Sometimes these are TSVs or have different encodings
    try:
        df = pd.read_csv(sample_file, lineterminator='\n')
        print("Success with lineterminator='\\n'")
        print(df.columns.tolist())
    except Exception as e2:
         print(f"Error reading with lineterminator: {e2}")
