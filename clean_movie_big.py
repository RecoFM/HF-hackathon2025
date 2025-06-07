import pandas as pd
import re

def analyze_and_clean_movies(input_path: str, output_path: str):
    """Analyze and clean movie titles, showing statistics for each cleaning reason."""
    
    # Load data
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"\nOriginal number of entries: {original_count:,}")
    
    # Check format patterns
    patterns = {
        "4K/UHD": r"\b(4K|UHD)\b",
        "Resolution (480p/720p/1080p)": r"\b[0-9]{3,4}p\b",
        "DVD/Blu-ray/VHS": r"\b(DVD|Blu[- ]?Ray|VHS)\b",
        "Rips (DVDRip/BRRip/etc)": r"\b(DVDRip|BRRip|HDRip|WEBRip)\b",
        "Video formats": r"\.(avi|mkv|mp4|mov)\b",
        "Collections": r"\b(Collection|Collector)\b"
    }
    
    print("\nFormat pattern matches:")
    print("=" * 50)
    for name, pattern in patterns.items():
        count = df["title"].str.contains(pattern, case=False, regex=True, na=False).sum()
        print(f"{name}: {count:,} titles")
        if count > 0:
            examples = df[df["title"].str.contains(pattern, case=False, regex=True, na=False)]["title"].head(2)
            for ex in examples:
                print(f"  Example: {ex}")
    
    # Remove 4K/UHD entries
    df_no_4k = df[~df["title"].str.contains(r"\b(4K|UHD)\b", case=False, na=False)].copy()
    removed_4k = original_count - len(df_no_4k)
    
    # Remove collections
    df_no_collections = df_no_4k[~df_no_4k["title"].str.contains(r"\b(Collection|Collector)\b", case=False, na=False)].copy()
    removed_collections = len(df_no_4k) - len(df_no_collections)
    
    # Clean titles (remove format info)
    def basic_clean(title):
        # Remove anything in brackets
        title = re.sub(r"\[.*?\]|\(.*?\)", "", title)
        # Remove format indicators
        title = re.sub(r"\b(DVD|Blu[- ]?Ray|VHS|[0-9]{3,4}p|4K|UHD)\b", "", title, flags=re.IGNORECASE)
        # Remove file extensions
        title = re.sub(r"\.(avi|mkv|mp4|mov)$", "", title, flags=re.IGNORECASE)
        # Clean up whitespace
        return re.sub(r"\s+", " ", title).strip()
    
    df_no_collections["clean_title"] = df_no_collections["title"].apply(basic_clean)
    
    # Count duplicates
    duplicates = df_no_collections["clean_title"].value_counts()
    duplicates = duplicates[duplicates > 1]
    
    print("\nDuplicate Analysis:")
    print("=" * 50)
    print(f"Titles with duplicates: {len(duplicates):,}")
    print(f"Total duplicate entries: {sum(duplicates) - len(duplicates):,}")
    
    if len(duplicates) > 0:
        print("\nTop duplicate examples:")
        for title, count in duplicates.head(3).items():
            print(f"\n'{title}' appears {count} times. Variants:")
            variants = df_no_collections[df_no_collections["clean_title"] == title]["title"].head(3)
            for v in variants:
                print(f"  - {v}")
    
    # Remove duplicates
    df_final = df_no_collections.drop_duplicates(subset="clean_title", keep="first")
    removed_duplicates = len(df_no_collections) - len(df_final)
    
    print("\nFinal Statistics:")
    print("=" * 50)
    print(f"Removed 4K/UHD versions: {removed_4k:,}")
    print(f"Removed collections: {removed_collections:,}")
    print(f"Removed duplicates: {removed_duplicates:,}")
    print(f"Total removed: {original_count - len(df_final):,}")
    print(f"Final count: {len(df_final):,}")
    print(f"Reduction: {((original_count - len(df_final)) / original_count * 100):.1f}%")
    
    # Save cleaned dataset
    df_final["title"] = df_final["clean_title"]
    df_final.drop(columns=["clean_title"], inplace=True)
    df_final.to_csv(output_path, index=False)

if __name__ == "__main__":
    analyze_and_clean_movies(
        input_path="amazon_movies_2023/title_embeddings_mapping.csv",
        output_path="amazon_movies_2023/title_embeddings_mapping_cleaned.csv"
    )
