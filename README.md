# Amazon Movies Dataset Processor

This script downloads and processes the Amazon Movies and TV 5-core dataset to create a user-item interaction matrix.

## Dataset

The script uses the Amazon Movies and TV 5-core dataset, which contains movie and TV show reviews from Amazon. The 5-core dataset means that all users and items have at least 5 reviews.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Simply run the script:

```bash
python process_amazon_dataset.py
```

The script will:
1. Download the dataset if it's not already present
2. Parse the gzipped JSON file
3. Create a user-item interaction matrix
4. Save the following files:
   - `user_mapping.npy`: Dictionary mapping user IDs to matrix indices
   - `item_mapping.npy`: Dictionary mapping item IDs to matrix indices
   - `interaction_matrix.npy`: The full user-item interaction matrix

## Output

The script will display statistics about the dataset, including:
- Number of users
- Number of items
- Number of ratings
- Matrix sparsity

The interaction matrix contains rating values from 1 to 5, representing user ratings for movies/TV shows. 