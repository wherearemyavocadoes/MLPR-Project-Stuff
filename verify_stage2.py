import pandas as pd

OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_text_dual.csv"

def verify():
    df = pd.read_csv(OUTPUT_FILE)
    print("Columns:", df.columns.tolist())
    print("\nNull counts:")
    print(df.isnull().sum())
    
    # Show random samples that likely had formatting
    sample_df = df.sample(3, random_state=42)
    for idx, row in sample_df.iterrows():
        print(f"\n--- Post {idx} ---")
        print(f"Title: {row.get('title', '')}")
        print(f"Selftext: {row.get('selftext', '')}")
        print(f"Stream A (Raw): {row.get('text_raw', '')}")
        print(f"Stream B (Clean): {row.get('text_clean', '')}")

if __name__ == "__main__":
    verify()
