import pandas as pd

OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_features.csv"

def verify():
    df = pd.read_csv(OUTPUT_FILE)
    print("\nTotal posts:", len(df))
    print("Total columns:", len(df.columns))
    print(df.columns.tolist()[:15], "... [And 384 embedding columns]")
    
    # Check random sample
    print("\n--- SAMPLE POST FEATURES ---")
    sample_df = df.sample(1, random_state=42)
    for col in sample_df.columns[:25]:
        print(f"{col}: {sample_df[col].values[0]}")

if __name__ == "__main__":
    verify()
