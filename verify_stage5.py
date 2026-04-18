import pandas as pd

OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_modeling_ready.csv"

def verify():
    df = pd.read_csv(OUTPUT_FILE)
    print("\n--- FINAL PREDICTIVE DATASET ---")
    print(f"Total user windows: {len(df)}")
    
    # Class Distribution
    print("\nTarget Label Distribution:")
    print(df['label'].value_counts())
    
    pct_positive = (df['label'].sum() / len(df)) * 100
    print(f"Positive Class Percentage: {pct_positive:.2f}%")
    
    # Show a positive example
    print("\n--- SAMPLE POSITIVE (PRE-CRISIS) WINDOW ---")
    positive_df = df[df['label'] == 1]
    if not positive_df.empty:
        sample = positive_df.sample(1, random_state=42)
        print(f"Author: {sample['author'].values[0]}")
        print(f"Window Start: {sample['window_start_time'].values[0]}")
        print(f"Window End: {sample['window_end_time'].values[0]}")
        print(f"Days to Crisis: {sample['days_to_crisis'].values[0]:.2f} days")
        print(f"Label: {sample['label'].values[0]}")
    else:
        print("No positive labels found.")

if __name__ == "__main__":
    verify()
