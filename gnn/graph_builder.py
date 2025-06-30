import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools

class GraphBuilder:
    """Simplified graph builder with core functionality"""
    
    def __init__(self,
                 method: str = 'correlation',
                 threshold: float = 0.3):
        self.method = method
        self.threshold = threshold

    def build_graph(self, feature_sequence: torch.Tensor) -> torch.LongTensor:
        """Build temporal graph from feature sequence"""
        if feature_sequence.ndim != 2:
            raise ValueError(f"Input must be 2D (time_steps, features), got shape {feature_sequence.shape}")
            
        T, F = feature_sequence.shape
        device = feature_sequence.device
        
        # Compute similarity matrix between time steps
        similarity_matrix = self._compute_similarity(feature_sequence)
        
        # Build edges
        return self._create_edges(similarity_matrix, T, device)

    def _compute_similarity(self, data: torch.Tensor) -> torch.Tensor:
        """Compute correlation between time steps"""
        T, F = data.shape
        centered = data - torch.mean(data, dim=1, keepdim=True)
        cov_matrix = torch.mm(centered, centered.t()) / (F - 1)
        
        stds = torch.std(data, dim=1)
        min_std = 1e-4
        stds = torch.clamp(stds, min=min_std)
        std_products = torch.outer(stds, stds)
        
        corr = cov_matrix / std_products
        return torch.clamp(corr, -1.0, 1.0)

    def _create_edges(self, matrix: torch.Tensor, 
                     num_nodes: int, device: torch.device) -> torch.LongTensor:
        """Create edge connections with fixed threshold"""
        # Get upper triangle indices
        indices = torch.triu_indices(num_nodes, num_nodes, 1, device=device)
        i, j = indices[0], indices[1]
        similarities = matrix[i, j]
        
        # Filter by threshold
        valid_mask = (torch.abs(similarities) > self.threshold)
        valid_i = i[valid_mask]
        valid_j = j[valid_mask]
        
        # Create bidirectional edges
        if valid_i.numel() > 0:
            edges = torch.stack([
                torch.cat([valid_i, valid_j]),
                torch.cat([valid_j, valid_i])
            ], dim=0)
        else:
            # Fallback to fully connected graph
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append([i, j])
            edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edges
