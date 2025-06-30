import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Union, List, Optional
import itertools

class GraphBuilder:
    """
    Builds dynamic correlation graphs from EMG time-series data with PyTorch support.
    Features:
    - Batch-aware graph construction
    - Multiple similarity metrics
    - Adaptive thresholding
    - Optimized small-sequence handling
    - Comprehensive validation

    Args:
        method: Similarity metric ('correlation', 'covariance', 'euclidean')
        threshold_type: 'fixed' or 'adaptive'
        default_threshold: Default threshold for fixed method
        adaptive_factor: Multiplier for adaptive threshold
        fully_connected_fallback: Use fully connected graph when no edges found
    """
    
    def __init__(self,
                 method: str = 'correlation',
                 threshold_type: str = 'adaptive',
                 default_threshold: float = 0.3,
                 adaptive_factor: float = 1.5,
                 fully_connected_fallback: bool = True):
        self.method = method
        self.threshold_type = threshold_type
        self.default_threshold = default_threshold
        self.adaptive_factor = adaptive_factor
        self.fully_connected_fallback = fully_connected_fallback
        
        if method not in {'correlation', 'covariance', 'euclidean'}:
            raise ValueError(f"Invalid method '{method}'. Choose from 'correlation', 'covariance', or 'euclidean'")
            
        if threshold_type not in {'fixed', 'adaptive'}:
            raise ValueError(f"Invalid threshold_type '{threshold_type}'. Choose 'fixed' or 'adaptive'")

    def build_graph(self, feature_sequence: Union[torch.Tensor, np.ndarray]) -> torch.LongTensor:
        """Build temporal graph from feature sequence"""
        # Convert numpy arrays to tensors
        if isinstance(feature_sequence, np.ndarray):
            feature_sequence = torch.from_numpy(feature_sequence).float()
        
        # Handle batch inputs using aggregated features
        if feature_sequence.ndim == 3:
            return self._build_batch_graph(feature_sequence)
            
        # Handle single sample
        elif feature_sequence.ndim == 2:
            return self._build_single_graph(feature_sequence)
            
        else:
            raise ValueError(f"Input must be 2D or 3D tensor, got shape {feature_sequence.shape}")

    def _build_batch_graph(self, batch_data: torch.Tensor) -> torch.LongTensor:
        """Build graph using aggregated batch features"""
        # Aggregate features across batch (weighted by variance)
        with torch.no_grad():
            variances = torch.var(batch_data, dim=(0, 2))
            weights = F.softmax(variances, dim=0)
            weighted_mean = torch.einsum('btd,d->td', batch_data, weights)
        
        return self._build_single_graph(weighted_mean)

    def _build_single_graph(self, feature_sequence: torch.Tensor) -> torch.LongTensor:
        """Build graph for a single sample"""
        if not isinstance(feature_sequence, torch.Tensor):
            raise TypeError(f"Input must be torch.Tensor, got {type(feature_sequence)}")
            
        if feature_sequence.ndim != 2:
            raise ValueError(f"Input must be 2D (time_steps, features), got shape {feature_sequence.shape}")
            
        T, F = feature_sequence.shape
        device = feature_sequence.device
        
        # Handle small sequences with optimized topology
        if T < 8:
            return self._create_optimized_topology(T).to(device)

        # Compute similarity matrix between time steps
        similarity_matrix = self._compute_similarity(feature_sequence)
        
        # Determine threshold
        threshold = self._determine_threshold(similarity_matrix)
        
        # Build edges
        return self._create_edges(similarity_matrix, threshold, T, device)

    def _compute_similarity(self, data: torch.Tensor) -> torch.Tensor:
        """Compute similarity between time steps"""
        T, F = data.shape
        
        if self.method == 'correlation':
            # Compute row-wise std with regularization
            stds = torch.std(data, dim=1)
            min_std = 1e-4
            stds = torch.clamp(stds, min=min_std)
            
            centered = data - torch.mean(data, dim=1, keepdim=True)
            cov_matrix = torch.mm(centered, centered.t()) / (F - 1)
            std_products = torch.outer(stds, stds)
            corr = cov_matrix / std_products
            return torch.clamp(corr, -1.0, 1.0)
            
        elif self.method == 'covariance':
            centered = data - torch.mean(data, dim=1, keepdim=True)
            cov = torch.mm(centered, centered.t()) / (F - 1)
            return torch.nan_to_num(cov, nan=0.0)
            
        elif self.method == 'euclidean':
            dist_matrix = torch.cdist(data, data, p=2)
            max_dist = torch.max(dist_matrix)
            min_dist = torch.min(dist_matrix[dist_matrix > 0]) if torch.any(dist_matrix > 0) else 1e-5
            normalized = (dist_matrix - min_dist) / (max_dist - min_dist + 1e-8)
            return 1 - torch.clamp(normalized, 0, 1)

    def _determine_threshold(self, matrix: torch.Tensor) -> float:
        """Calculate appropriate threshold"""
        if self.threshold_type == 'fixed':
            return self.default_threshold
        
        # Adaptive threshold based on median absolute similarity
        abs_matrix = torch.abs(matrix)
        abs_matrix.fill_diagonal_(0)  # Ignore self-connections
        
        # Flatten and remove zeros
        flat_matrix = abs_matrix.flatten()
        non_zero = flat_matrix[flat_matrix > 0]
        
        if non_zero.numel() == 0:
            return 0.0
            
        median_val = torch.median(non_zero).item()
        return median_val * self.adaptive_factor

    def _create_edges(self, matrix: torch.Tensor, threshold: float, 
                     num_nodes: int, device: torch.device) -> torch.LongTensor:
        """Create edge connections with validation"""
        # Get upper triangle indices
        indices = torch.triu_indices(num_nodes, num_nodes, 1, device=device)
        i, j = indices[0], indices[1]
        similarities = matrix[i, j]
        
        # Filter by threshold and valid indices
        valid_mask = (torch.abs(similarities) > threshold) & (i < num_nodes) & (j < num_nodes)
        valid_i = i[valid_mask]
        valid_j = j[valid_mask]
        
        # Create bidirectional edges
        if valid_i.numel() > 0:
            edges = torch.stack([
                torch.cat([valid_i, valid_j]),
                torch.cat([valid_j, valid_i])
            ], dim=0)
        else:
            edges = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Handle no-edge case
        if edges.numel() == 0 and self.fully_connected_fallback:
            return self._create_optimized_topology(num_nodes).to(device)
        
        return edges

    def _create_optimized_topology(self, num_nodes: int) -> torch.LongTensor:
        """Create optimized topology for small sequences"""
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long)
        
        edges = []
        # Create bidirectional chain connections
        for i in range(num_nodes - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])
            
        # Add skip connections for sequences > 3
        if num_nodes > 3:
            for i in range(num_nodes - 2):
                edges.append([i, i+2])
                edges.append([i+2, i])
                
        # Add cross connections for sequences > 5
        if num_nodes > 5:
            mid = num_nodes // 2
            for i in range(mid):
                if i != mid:
                    edges.append([i, mid])
                    edges.append([mid, i])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def build_graph_for_batch(self, batch_data: torch.Tensor) -> List[torch.LongTensor]:
        """
        Build graphs for each sample in a batch
        
        Args:
            batch_data: Tensor of shape (batch, time_steps, features)
            
        Returns:
            List of edge_index tensors for each sample
        """
        if batch_data.ndim != 3:
            raise ValueError(f"Batch input must be 3D, got shape {batch_data.shape}")
            
        edge_indices = []
        for i in range(batch_data.size(0)):
            edge_index = self._build_single_graph(batch_data[i])
            edge_indices.append(edge_index)
            
        return edge_indices
