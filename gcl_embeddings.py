from typing import Dict, Any, Optional, List
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.preprocessing import normalize
import os
import pandas as pd

class GCLProcessor:
    def __init__(self, data_dir: str = "amazon_movies_2023", n_layers: int = 1) -> None:
        self.data_dir: str = data_dir
        self.n_layers: int = n_layers
        self.original_embeddings: Optional[NDArray[np.float32]] = None
        self.item_ids: Optional[NDArray[np.str_]] = None
        self.user_item_matrix: Optional[sp.spmatrix] = None
        self.adj_matrix: Optional[sp.spmatrix] = None
        self.embedding_item_mapping: Optional[Dict[str, int]] = None
        self.matrix_item_mapping: Optional[Dict[str, int]] = None
        
    def load_data(self) -> None:
        """Load embeddings and user-item interaction matrix"""
        print("Loading data...")
        
        # Load LLM embeddings
        embeddings_path: str = os.path.join(self.data_dir, "title_embeddings.npz")
        embeddings_data: Dict[str, NDArray] = np.load(embeddings_path)
        self.original_embeddings = embeddings_data['embeddings'].astype(np.float32)
        self.item_ids = embeddings_data['item_ids']
        
        # Create embedding item mapping
        self.embedding_item_mapping = {str(item_id): idx for idx, item_id in enumerate(self.item_ids)}
        
        # Load user-item matrix
        matrix_path: str = os.path.join(self.data_dir, "user_item_matrix.npz")
        self.user_item_matrix = sp.load_npz(matrix_path)
        
        # Load matrix item mapping
        mapping_path: str = os.path.join(self.data_dir, "item_id_mapping.csv")
        item_mapping_df = pd.read_csv(mapping_path)
        self.matrix_item_mapping = dict(zip(item_mapping_df['item_id'].astype(str), item_mapping_df.index))
        
        print(f"Loaded embeddings shape: {self.original_embeddings.shape}")
        print(f"Loaded user-item matrix shape: {self.user_item_matrix.shape}")
        
        # Align embeddings with matrix dimensions
        self._align_embeddings()
        
    def _align_embeddings(self) -> None:
        """Align embeddings with the user-item matrix dimensions"""
        if self.matrix_item_mapping is None or self.embedding_item_mapping is None:
            raise ValueError("Item mappings not initialized")
            
        n_items = self.user_item_matrix.shape[1]
        embedding_dim = self.original_embeddings.shape[1]
        
        # Create new aligned embeddings matrix
        aligned_embeddings = np.zeros((n_items, embedding_dim), dtype=np.float32)
        
        # Map embeddings to correct positions
        common_items = set(self.matrix_item_mapping.keys()) & set(self.embedding_item_mapping.keys())
        print(f"Found {len(common_items)} common items between matrix and embeddings")
        
        for item_id in common_items:
            matrix_idx = self.matrix_item_mapping[item_id]
            emb_idx = self.embedding_item_mapping[item_id]
            aligned_embeddings[matrix_idx] = self.original_embeddings[emb_idx]
            
        # Handle items without embeddings by using mean embedding
        mean_embedding = np.mean(self.original_embeddings, axis=0)
        zero_rows = np.where(~aligned_embeddings.any(axis=1))[0]
        aligned_embeddings[zero_rows] = mean_embedding
        
        self.original_embeddings = aligned_embeddings
        print(f"Aligned embeddings shape: {self.original_embeddings.shape}")
        
    def normalize_adj_matrix(self, adj: sp.spmatrix) -> sp.spmatrix:
        """Symmetrically normalize adjacency matrix"""
        rowsum: NDArray[np.float32] = np.array(adj.sum(1), dtype=np.float32)
        d_inv_sqrt: NDArray[np.float32] = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt: sp.spmatrix = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    def create_adjacency_matrix(self) -> None:
        """Create adjacency matrix from user-item interactions"""
        print("Creating adjacency matrix...")
        
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not loaded. Call load_data() first.")
        
        # Create item-item adjacency matrix through user interactions
        # A = R^T R where R is user-item matrix
        adj_matrix: sp.spmatrix = self.user_item_matrix.T.dot(self.user_item_matrix)
        
        # Normalize without adding self-loops
        self.adj_matrix = self.normalize_adj_matrix(adj_matrix)
        print(f"Created adjacency matrix shape: {self.adj_matrix.shape}")
    
    def gcn_layer(self, embeddings: NDArray[np.float32], adj_matrix: sp.spmatrix) -> NDArray[np.float32]:
        """Apply one GCN layer"""
        # Simple GCN layer without learnable parameters
        # Just propagating information through the graph
        return adj_matrix.dot(embeddings)
    
    def process_embeddings(self) -> NDArray[np.float32]:
        """Apply GCN layers and combine embeddings"""
        print(f"Processing embeddings through {self.n_layers} GCN layer(s)...")
        
        if self.original_embeddings is None or self.adj_matrix is None:
            raise ValueError("Embeddings or adjacency matrix not initialized. Call load_data() and create_adjacency_matrix() first.")
        
        # Normalize original embeddings
        embeddings: NDArray[np.float32] = normalize(self.original_embeddings)
        
        # List to store all embeddings (including original)
        all_embeddings: List[NDArray[np.float32]] = [embeddings]
        
        # Current embedding state
        current_embeddings: NDArray[np.float32] = embeddings
        
        # Apply GCN layers
        for i in range(self.n_layers):
            current_embeddings = self.gcn_layer(current_embeddings, self.adj_matrix)
            current_embeddings = normalize(current_embeddings)
            all_embeddings.append(current_embeddings)
        
        # Average all embeddings (original + GCL layers)
        final_embeddings: NDArray[np.float32] = np.mean(all_embeddings, axis=0)
        
        # Normalize final embeddings
        final_embeddings = normalize(final_embeddings)
        
        print("Embeddings processing complete")
        return final_embeddings
    
    def save_embeddings(self, final_embeddings: NDArray[np.float32]) -> None:
        """Save the processed embeddings"""
        if self.matrix_item_mapping is None:
            raise ValueError("Item mapping not loaded. Call load_data() first.")
            
        output_path: str = os.path.join(self.data_dir, "gcl_embeddings.npz")
        
        # Convert matrix indices back to item IDs
        reverse_mapping = {idx: item_id for item_id, idx in self.matrix_item_mapping.items()}
        item_ids = np.array([reverse_mapping[i] for i in range(len(reverse_mapping))])
        
        np.savez(
            output_path,
            embeddings=final_embeddings,
            item_ids=item_ids
        )
        print(f"Saved processed embeddings to {output_path}")

def main() -> None:
    # Initialize processor with 1 GCL layer
    processor: GCLProcessor = GCLProcessor(n_layers=1)
    
    # Load data
    processor.load_data()
    
    # Create adjacency matrix
    processor.create_adjacency_matrix()
    
    # Process embeddings through GCN layers
    final_embeddings: NDArray[np.float32] = processor.process_embeddings()
    
    # Save results
    processor.save_embeddings(final_embeddings)
    
if __name__ == "__main__":
    main() 