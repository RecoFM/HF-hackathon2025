import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_cleaning_results():
    # File paths
    original_file = "amazon_movies_2023/title_embeddings_mapping.csv"
    cleaned_file = "amazon_movies_2023/title_embeddings_mapping_cleaned.csv"
    
    # Check if files exist
    if not all(Path(f).exists() for f in [original_file, cleaned_file]):
        print("Error: One or both input files not found!")
        return
    
    # Read the datasets
    original_df = pd.read_csv(original_file)
    cleaned_df = pd.read_csv(cleaned_file)
    
    # Calculate statistics
    original_count = len(original_df)
    cleaned_count = len(cleaned_df)
    removed_count = original_count - cleaned_count
    removal_percentage = (removed_count / original_count) * 100
    
    # Print analysis results
    print("\nCleaning Analysis Results:")
    print("=" * 50)
    print(f"Original number of movies: {original_count:,}")
    print(f"Number of movies after cleaning: {cleaned_count:,}")
    print(f"Number of movies removed: {removed_count:,}")
    print(f"Percentage of movies removed: {removal_percentage:.2f}%")
    
    # Sample of removed titles
    if removed_count > 0:
        print("\nSample of removed titles:")
        print("-" * 50)
        removed_titles = set(original_df['title']) - set(cleaned_df['title'])
        for title in list(removed_titles)[:5]:  # Show first 5 removed titles
            print(f"- {title}")

def analyze_duplicates(input_path: str) -> None:
    """Analyze duplicates in the movie dataset."""
    # Load the CSV
    print("\nLoading data...")
    df = pd.read_csv(input_path)
    original_count = len(df)
    
    print("\nDuplicate Analysis:")
    print("=" * 50)
    print(f"Total number of entries: {original_count:,}")
    
    # Count exact duplicates
    exact_dups = df["title"].value_counts()
    exact_dups = exact_dups[exact_dups > 1]
    print(f"\nExact duplicates found: {len(exact_dups):,} titles")
    print(f"Total duplicate entries: {sum(exact_dups) - len(exact_dups):,}")
    
    if len(exact_dups) > 0:
        print("\nTop 5 most duplicated titles:")
        print("-" * 50)
        for title, count in exact_dups.head().items():
            print(f"'{title}' appears {count} times")
    
    # Count similar titles (case-insensitive)
    df["title_lower"] = df["title"].str.lower()
    case_dups = df["title_lower"].value_counts()
    case_dups = case_dups[case_dups > 1]
    print(f"\nCase-insensitive duplicates found: {len(case_dups):,} titles")
    
    if len(case_dups) > 0:
        print("\nTop 5 case-insensitive duplicates:")
        print("-" * 50)
        for title_lower, count in case_dups.head().items():
            variants = df[df["title_lower"] == title_lower]["title"].unique()
            print(f"\nTitle appears {count} times with variants:")
            for v in variants:
                print(f"  - {v}")

if __name__ == "__main__":
    analyze_cleaning_results()
    analyze_duplicates("amazon_movies_2023/title_embeddings_mapping.csv") 