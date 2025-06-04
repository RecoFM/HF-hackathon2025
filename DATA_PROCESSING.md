# Data Processing Pipeline

This document describes how we process the Amazon Movies and TV 5-core dataset to create the recommendation system's data files.

## Dataset

The system uses the Amazon Movies and TV 5-core dataset, which contains movie and TV show reviews from Amazon. The 5-core dataset means that all users and items have at least 5 reviews.

## Processing Steps

### 1. Initial Data Processing
The `process_amazon_dataset.py` script:
1. Downloads the dataset if not present
2. Parses the gzipped JSON file
3. Creates a user-item interaction matrix
4. Saves the following files:
   - `user_mapping.npy`: Dictionary mapping user IDs to matrix indices
   - `item_mapping.npy`: Dictionary mapping item IDs to matrix indices
   - `interaction_matrix.npy`: The full user-item interaction matrix

### 2. Title Embeddings Generation
The `create_title_embeddings.py` script:
1. Processes movie titles using Mistral AI's language model
2. Creates semantic embeddings for each title
3. Saves:
   - `title_embeddings.npz`: Raw LLM embeddings
   - `title_embeddings_mapping.csv`: Movie metadata

### 3. Graph-Enhanced Embeddings
The `gcl_embeddings.py` script:
1. Takes the LLM embeddings as input
2. Applies a Graph Convolutional Layer (GCL)
3. Combines semantic understanding with user interaction patterns
4. Saves:
   - `gcl_embeddings.npz`: Enhanced embeddings

## Statistics

The processed dataset includes:
- Number of movies with LLM embeddings: 113,573
- Number of movies with GCL embeddings: 115,101
- Number of movies with both types: 101,015
- Matrix sparsity: Varies by embedding type

## File Formats

### Embedding Files (.npz)
- Compressed NumPy arrays
- Contains:
  - `embeddings`: Shape (n_items, 1024) float32 array
  - `item_ids`: Shape (n_items,) array of item identifiers

### Mapping Files (.csv)
- Contains movie metadata:
  - `item_id`: Unique identifier
  - `title`: Movie/TV show title
  - Additional metadata fields

### Interaction Matrix
- Sparse matrix format
- Values: 1-5 (user ratings)
- Dimensions: (n_users, n_items) 