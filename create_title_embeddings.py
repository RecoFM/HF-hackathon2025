from typing import List, Dict, Any, Optional, Set
import os
import pandas as pd
import numpy as np
from mistralai import Mistral
from tqdm import tqdm
from dotenv import load_dotenv
import time
import json
from dataclasses import dataclass

class EmptyTitleError(Exception):
    """Exception raised when an item has an empty title."""
    pass

class RateLimitError(Exception):
    """Exception raised when hitting API rate limits."""
    pass

@dataclass
class Config:
    """Configuration for the embedding generation process"""
    input_dir: str = "amazon_movies_2023"
    output_dir: str = "amazon_movies_2023"
    batch_size: int = 256  # Reduced batch size to avoid hitting limits
    model_name: str = "mistral-embed"
    rate_limit_delay: float = 1.5  # Slightly more than 1 second to be safe
    input_file: str = "item_id_mapping.csv"
    output_embeddings: str = "title_embeddings.npz"
    output_mapping: str = "title_embeddings_mapping.csv"
    progress_file: str = "embedding_progress.json"

def load_item_data(file_path: str) -> pd.DataFrame:
    """
    Load item data from CSV file and check for empty titles
    
    Args:
        file_path: Path to the CSV file containing item data
        
    Returns:
        DataFrame containing items with titles
        
    Raises:
        EmptyTitleError: If any item has an empty title after removing whitespace
    """
    print(f"Loading item data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check for completely missing titles
    if df['title'].isna().any():
        missing_ids = df[df['title'].isna()]['item_id'].tolist()
        raise EmptyTitleError(f"Found {len(missing_ids)} items with missing titles. First few item IDs: {missing_ids[:5]}")
    
    # Check for empty strings or whitespace-only titles
    df['title'] = df['title'].str.strip()
    empty_titles = df[df['title'] == '']
    if not empty_titles.empty:
        empty_ids = empty_titles['item_id'].tolist()
        raise EmptyTitleError(f"Found {len(empty_ids)} items with empty titles (after stripping whitespace). First few item IDs: {empty_ids[:5]}")
    
    print(f"Found {len(df):,} items with valid titles")
    return df

def load_progress(config: Config) -> Set[int]:
    """Load the set of item IDs that already have embeddings"""
    progress_path = os.path.join(config.output_dir, config.progress_file)
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(completed_ids: Set[int], config: Config) -> None:
    """Save the set of completed item IDs"""
    progress_path = os.path.join(config.output_dir, config.progress_file)
    with open(progress_path, 'w') as f:
        json.dump(list(completed_ids), f)

def create_embeddings_batch(
    client: Mistral,
    texts: List[str],
    item_ids: List[int],
    completed_ids: Set[int],
    all_embeddings: Dict[int, List[float]],
    config: Config
) -> None:
    """
    Create embeddings for a batch of texts with rate limiting
    
    Args:
        client: Initialized Mistral client
        texts: List of texts to create embeddings for
        item_ids: List of corresponding item IDs
        completed_ids: Set of item IDs that already have embeddings
        all_embeddings: Dictionary to store embeddings
        config: Configuration object
        
    Raises:
        RateLimitError: If rate limit is exceeded
    """
    try:
        embeddings = client.embeddings.create(
            model=config.model_name,
            inputs=texts
        )
        
        # Store embeddings and update progress
        for item_id, emb in zip(item_ids, embeddings.data):
            all_embeddings[item_id] = emb.embedding
            completed_ids.add(item_id)
        
        # Save progress after each successful batch
        save_progress(completed_ids, config)
        
        # Rate limiting
        time.sleep(config.rate_limit_delay)
        
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise RateLimitError(f"Rate limit exceeded: {str(e)}")
        raise

def save_embeddings(
    embeddings_dict: Dict[int, List[float]],
    df: pd.DataFrame,
    config: Config
) -> None:
    """
    Save embeddings and mapping files
    
    Args:
        embeddings_dict: Dictionary mapping item IDs to embeddings
        df: DataFrame with item data
        config: Configuration object
    """
    # Convert dictionary to arrays
    item_ids = np.array(list(embeddings_dict.keys()))
    embeddings = np.array([embeddings_dict[id_] for id_ in item_ids])
    
    # Save embeddings
    output_path = os.path.join(config.output_dir, config.output_embeddings)
    print(f"\nSaving embeddings to {output_path}")
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        item_ids=item_ids
    )
    
    # Save mapping
    mapping_path = os.path.join(config.output_dir, config.output_mapping)
    df_completed = df[df['item_id'].isin(item_ids)]
    df_completed.to_csv(mapping_path, index=False)
    print(f"Saved mapping to {mapping_path}")
    print(f"\nTotal embeddings created: {len(item_ids):,}")

def main() -> None:
    # Load environment variables (for MISTRAL_API_KEY)
    load_dotenv()
    
    # Initialize configuration
    config = Config()
    
    # Check for API key
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    # Load item data
    input_path = os.path.join(config.input_dir, config.input_file)
    df = load_item_data(input_path)
    
    # Load progress
    completed_ids = load_progress(config)
    print(f"\nFound {len(completed_ids):,} previously completed embeddings")
    
    # Filter out completed items
    df_remaining = df[~df['item_id'].isin(completed_ids)]
    print(f"Remaining items to process: {len(df_remaining):,}")
    
    if len(df_remaining) == 0:
        print("All items already processed!")
        return
    
    # Initialize embeddings dictionary with any existing progress
    all_embeddings: Dict[int, List[float]] = {}
    
    # Process remaining items in batches
    try:
        for i in tqdm(range(0, len(df_remaining), config.batch_size), desc="Creating embeddings"):
            batch_df = df_remaining.iloc[i:i + config.batch_size]
            create_embeddings_batch(
                client,
                batch_df['title'].tolist(),
                batch_df['item_id'].tolist(),
                completed_ids,
                all_embeddings,
                config
            )
    except RateLimitError as e:
        print(f"\n⚠️ {str(e)}")
        print("Saving current progress...")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        print("Saving current progress...")
    finally:
        if all_embeddings:
            save_embeddings(all_embeddings, df, config)
            print("\nProgress has been saved. You can resume later by running the script again.")
        
        print(f"\nCompleted embeddings: {len(completed_ids):,} out of {len(df):,} ({len(completed_ids)/len(df):.1%})")

if __name__ == "__main__":
    main() 