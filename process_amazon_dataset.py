from typing import Dict, List, Any, Tuple
from datasets import load_dataset, Dataset
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import os
from dataclasses import dataclass, field

@dataclass
class Config:
    """Configuration for Amazon dataset processing"""
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    reviews_config: str = "5core_rating_only_Movies_and_TV"
    metadata_config: str = "raw_meta_Movies_and_TV"
    split: str = "full"
    output_dir: str = "amazon_movies_2023"
    required_review_columns: List[str] = field(default_factory=lambda: ["user_id", "parent_asin", "rating"])
    required_meta_columns: List[str] = field(default_factory=lambda: ["parent_asin", "title"])
    min_item_interactions: int = 5  # Minimum number of interactions per item

def load_datasets(config: Config) -> Tuple[Dataset, Dataset]:
    """
    Load reviews and metadata datasets
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (reviews dataset, metadata dataset)
    """
    print("Downloading datasets...")
    reviews = load_dataset(
        config.dataset_name,
        name=config.reviews_config,
        split=config.split,
        trust_remote_code=True
    )
    metadata = load_dataset(
        config.dataset_name,
        name=config.metadata_config,
        split=config.split,
        trust_remote_code=True
    )
    return reviews, metadata

def process_metadata(df_meta: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Process metadata DataFrame and filter out items with empty titles
    
    Args:
        df_meta: Raw metadata DataFrame
        config: Configuration object
        
    Returns:
        Processed metadata DataFrame
    """
    print("\nProcessing metadata...")
    initial_items = len(df_meta)
    
    # Remove items with missing titles
    missing_titles = df_meta['title'].isna().sum()
    df_meta = df_meta[config.required_meta_columns].dropna()
    
    # Remove items with empty or whitespace-only titles
    df_meta['title'] = df_meta['title'].str.strip()
    empty_titles = (df_meta['title'] == '').sum()
    df_meta = df_meta[df_meta['title'] != '']
    
    # Remove duplicates
    df_meta = df_meta.drop_duplicates(subset=["parent_asin"])
    
    # Print statistics
    total_removed = initial_items - len(df_meta)
    print(f"\nMetadata filtering statistics:")
    print(f"Initial items: {initial_items:,}")
    print(f"Items with missing titles: {missing_titles:,} ({missing_titles/initial_items:.1%})")
    print(f"Items with empty titles: {empty_titles:,} ({empty_titles/initial_items:.1%})")
    print(f"Total items removed: {total_removed:,} ({total_removed/initial_items:.1%})")
    print(f"Remaining items: {len(df_meta):,} ({len(df_meta)/initial_items:.1%})")
    
    return df_meta

def process_reviews(df_reviews: pd.DataFrame, df_meta: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Process reviews DataFrame and filter items with few interactions
    
    Args:
        df_reviews: Raw reviews DataFrame
        df_meta: Processed metadata DataFrame with valid titles
        config: Configuration object
        
    Returns:
        Processed reviews DataFrame
    """
    print("\nProcessing reviews...")
    initial_reviews = len(df_reviews)
    initial_items = df_reviews['parent_asin'].nunique()
    
    # Basic filtering
    df_reviews = df_reviews[config.required_review_columns].dropna()
    df_reviews = df_reviews.drop_duplicates(subset=["user_id", "parent_asin"])
    df_reviews["rating"] = df_reviews["rating"].astype(float)
    
    # Keep only reviews for items that have titles
    items_with_titles = set(df_meta['parent_asin'])
    df_reviews = df_reviews[df_reviews['parent_asin'].isin(items_with_titles)]
    
    # Count item interactions and filter low-interaction items
    item_counts = df_reviews['parent_asin'].value_counts()
    popular_items = item_counts[item_counts >= config.min_item_interactions].index
    df_reviews = df_reviews[df_reviews['parent_asin'].isin(popular_items)]
    
    # Print statistics
    final_reviews = len(df_reviews)
    final_items = df_reviews['parent_asin'].nunique()
    
    print(f"\nReview filtering statistics:")
    print(f"Initial reviews: {initial_reviews:,}")
    print(f"Initial items: {initial_items:,}")
    print(f"Items without titles removed: {initial_items - len(items_with_titles):,}")
    print(f"Items with < {config.min_item_interactions} interactions: {len(items_with_titles) - final_items:,}")
    print(f"Reviews removed: {initial_reviews - final_reviews:,} ({(initial_reviews - final_reviews)/initial_reviews:.1%})")
    print(f"Remaining reviews: {final_reviews:,} ({final_reviews/initial_reviews:.1%})")
    print(f"Remaining items: {final_items:,} ({final_items/initial_items:.1%})")
    
    return df_reviews

def create_id_mappings(df_reviews: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    """
    Create user and item ID mappings
    
    Args:
        df_reviews: Processed reviews DataFrame
        
    Returns:
        Tuple of (user2id, item2id, id2user, id2item) mappings
    """
    user_ids = df_reviews["user_id"].astype("category").cat
    item_ids = df_reviews["parent_asin"].astype("category").cat

    user2id = dict(zip(user_ids.categories, range(len(user_ids.categories))))
    item2id = dict(zip(item_ids.categories, range(len(item_ids.categories))))
    id2user = dict(enumerate(user_ids.categories))
    id2item = dict(enumerate(item_ids.categories))

    print(f"\nTotal unique users: {len(user2id):,}")
    print(f"Total unique items in reviews: {len(item2id):,}")
    
    return user2id, item2id, id2user, id2item

def build_interaction_matrix(
    df_reviews: pd.DataFrame,
    user2id: Dict[str, int],
    item2id: Dict[str, int]
) -> csr_matrix:
    """
    Build sparse interaction matrix
    
    Args:
        df_reviews: Processed reviews DataFrame
        user2id: User ID mapping
        item2id: Item ID mapping
        
    Returns:
        Sparse interaction matrix
    """
    print("\nBuilding interaction matrix...")
    user_ids = df_reviews["user_id"].astype("category").cat
    item_ids = df_reviews["parent_asin"].astype("category").cat
    
    rows = user_ids.codes
    cols = item_ids.codes
    data = df_reviews["rating"].values
    
    return csr_matrix((data, (rows, cols)), shape=(len(user2id), len(item2id)))

def save_results(
    user2id: Dict[str, int],
    item2id: Dict[str, int],
    df_meta: pd.DataFrame,
    interaction_matrix: csr_matrix,
    config: Config
) -> None:
    """
    Save all processed data
    
    Args:
        user2id: User ID mapping
        item2id: Item ID mapping
        df_meta: Processed metadata DataFrame
        interaction_matrix: Interaction matrix
        config: Configuration object
    """
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"\nSaving data to '{config.output_dir}/'...")

    # Save user ID mapping
    pd.DataFrame({
        "user_id": list(user2id.values()),
        "original_user_id": list(user2id.keys())
    }).to_csv(f"{config.output_dir}/user_id_mapping.csv", index=False)

    # Save item ID mapping with titles
    item_mapping = pd.DataFrame({
        "item_id": list(item2id.values()),
        "parent_asin": list(item2id.keys())
    })
    item_mapping = item_mapping.merge(df_meta, on="parent_asin", how="left")
    
    # Print title coverage statistics
    items_with_titles = item_mapping["title"].notna().sum()
    print(f"\nItems with titles: {items_with_titles:,} out of {len(item_mapping):,} ({items_with_titles/len(item_mapping):.1%})")
    
    item_mapping.to_csv(f"{config.output_dir}/item_id_mapping.csv", index=False)

    # Save interaction matrix
    save_npz(f"{config.output_dir}/user_item_matrix.npz", interaction_matrix)

def print_statistics(interaction_matrix: csr_matrix, config: Config) -> None:
    """
    Print final statistics
    
    Args:
        interaction_matrix: Processed interaction matrix
        config: Configuration object
    """
    print("\n✅ Processing complete!")
    print(f"Matrix shape: {interaction_matrix.shape}")
    print(f"Number of interactions: {interaction_matrix.nnz:,}")
    print(f"Sparsity: {interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.4%}")
    print(f"\nFiles saved in '{config.output_dir}/' directory:")
    print("- user_id_mapping.csv")
    print("- item_id_mapping.csv (includes titles)")
    print("- user_item_matrix.npz")

def main() -> None:
    # Initialize configuration
    config = Config()
    
    # Load datasets
    reviews, metadata = load_datasets(config)
    
    # Convert to pandas DataFrames
    print("Converting to DataFrames...")
    df_reviews = reviews.to_pandas()
    df_meta = metadata.to_pandas()

    # Print initial statistics
    print(f"\nReviews dataset size: {len(df_reviews):,} reviews")
    print(f"Metadata dataset size: {len(df_meta):,} items")
    print("\nReviews columns:")
    print(df_reviews.columns.tolist())
    print("\nMetadata columns:")
    print(df_meta.columns.tolist())

    # Process metadata first to get valid items
    df_meta = process_metadata(df_meta, config)
    
    # Then process reviews, keeping only items with valid titles
    df_reviews = process_reviews(df_reviews, df_meta, config)
    
    # Print rating statistics
    print("\nRating statistics:")
    print(df_reviews["rating"].describe())

    # Create mappings and build matrix
    user2id, item2id, id2user, id2item = create_id_mappings(df_reviews)
    interaction_matrix = build_interaction_matrix(df_reviews, user2id, item2id)
    
    # Save results and print statistics
    save_results(user2id, item2id, df_meta, interaction_matrix, config)
    print_statistics(interaction_matrix, config)

if __name__ == "__main__":
    main() 