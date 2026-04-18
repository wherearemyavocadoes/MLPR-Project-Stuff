import pandas as pd

OUTPUT_FILE = "/Users/arya_vachhani/Downloads/Reddit Data/processed_2019_user_temporal.csv"

def verify():
    df = pd.read_csv(OUTPUT_FILE)
    print("Columns:", df.columns.tolist())
    print(f"Total rows: {len(df)}")
    
    unique_users = df['author'].nunique()
    print(f"Unique users: {unique_users}")
    
    # Check nulls
    print("\nNull counts:")
    print(df.isnull().sum())
    
    # Show top 2 users' posts
    print("\nSample user grouping:")
    top_users = df['author'].value_counts().head(2).index.tolist()
    
    for u in top_users:
        user_posts = df[df['author'] == u]
        print(f"\nUser: {u} (Total posts: {len(user_posts)})")
        print(user_posts[['timestamp', 'subreddit', 'title']].head(5))

if __name__ == "__main__":
    verify()
