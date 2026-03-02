import pandas as pd

OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_temporal_windows.csv"

def verify():
    df = pd.read_csv(OUTPUT_FILE)
    print("\nTotal user windows:", len(df))
    print("Total columns:", len(df.columns))
    
    # We drop the heavy embeddings just for printing
    print_cols = [c for c in df.columns if not c.startswith('win_emb_')]
    print(print_cols)
    
    # Check random sample
    print("\n--- SAMPLE TEMPORAL WINDOW ---")
    sample_df = df[print_cols].sample(1, random_state=12)
    for col in print_cols:
        val = sample_df[col].values[0]
        if isinstance(val, float):
             print(f"{col}: {val:.4f}")
        else:
             print(f"{col}: {val}")

if __name__ == "__main__":
    verify()
