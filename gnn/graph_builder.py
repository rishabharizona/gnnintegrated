import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Union, List
import itertools

class GraphBuilder:
    """
    Builds dynamic correlation graphs from EMG time-series data with multiple
    connectivity strategies and adaptive thresholding for biomedical research.
    
    Features:
    - Multiple similarity metrics (correlation, covariance, Euclidean distance)
    - Adaptive thresholding based on data distribution
    - Edge weight preservation
    - Batch processing support
    - Comprehensive validation checks
    
    Args:
        method: Similarity metric ('correlation', 'covariance', 'euclidean')
        threshold_type: 'fixed' or 'adaptive' (median-based)
        default_threshold: Default threshold value for fixed method
        adaptive_factor: Multiplier for adaptive threshold calculation
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

    def build_graph(self, feature_sequence: np.ndarray) -> torch.LongTensor:
        """
        Build temporal graph from feature sequence (post-convolution)
        
        Args:
            feature_sequence: Feature array of shape (time_steps, features)
            
        Returns:
            edge_index: Tensor of shape [2, num_edges]
        """
        # Validate input
        if not isinstance(feature_sequence, np.ndarray):
            raise TypeError(f"Input must be numpy array, got {type(feature_sequence)}")
            
        if feature_sequence.ndim != 2:
            raise ValueError(f"Input must be 2D (time_steps, features), got shape {feature_sequence.shape}")
            
        T, F = feature_sequence.shape
        if T < 10:
            print(f"⚠️ Insufficient time steps ({T}), using fully connected graph")
            return self._build_fully_connected(T)

        # Compute similarity matrix between TIME STEPS
        similarity_matrix = self._compute_similarity(feature_sequence)
        
        # Determine threshold
        threshold = self._determine_threshold(similarity_matrix)
        
        # Build edges using TIME STEPS as nodes
        return self._create_edges(similarity_matrix, threshold, T)

    def _compute_similarity(self, data: np.ndarray) -> np.ndarray:
        """Compute similarity between time steps (temporal correlation)"""
        T, F = data.shape
        
        if self.method == 'correlation':
            # Compute row-wise (time steps) std
            stds = np.std(data, axis=1)
            constant_mask = stds < 1e-8
            if np.any(constant_mask):
                # Add small noise to constant time steps
                noise = np.random.normal(0, 1e-8, data.shape)
                data = data + noise * constant_mask[:, np.newaxis]
            
            # Compute time-step correlation
            cov_matrix = np.cov(data, rowvar=True)
            std_products = np.outer(stds, stds)
            std_products[std_products < 1e-10] = 1e-10
            corr = cov_matrix / std_products
            np.clip(corr, -1.0, 1.0, out=corr)
            return corr
            
        elif self.method == 'covariance':
            cov = np.cov(data, rowvar=True)
            np.nan_to_num(cov, copy=False, nan=0.0)
            return cov
            
        elif self.method == 'euclidean':
            dist_matrix = squareform(pdist(data, 'euclidean'))
            max_dist = np.max(dist_matrix)
            if max_dist < 1e-8:
                return np.ones_like(dist_matrix)
            similarity = 1 - (dist_matrix / max_dist)
            np.clip(similarity, -1.0, 1.0, out=similarity)
            return similarity

    def _determine_threshold(self, matrix: np.ndarray) -> float:
        """Calculate appropriate threshold based on type"""
        if self.threshold_type == 'fixed':
            return self.default_threshold
        
        # Adaptive threshold based on median absolute similarity
        abs_matrix = np.abs(matrix)
        np.fill_diagonal(abs_matrix, 0)  # Ignore self-connections
        non_zero_elements = abs_matrix[abs_matrix > 0]
        
        # Handle case where all similarities are zero
        if non_zero_elements.size == 0:
            return 0.0
            
        median_val = np.median(non_zero_elements)
        return median_val * self.adaptive_factor

    def _create_edges(self, matrix: np.ndarray, threshold: float, num_nodes: int) -> torch.LongTensor:
        """Create edge connections between TIME STEPS (nodes)"""
        edges = []
        
        # Create edges where absolute similarity exceeds threshold
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):  # Only upper triangle
                similarity = matrix[i, j]
                if abs(similarity) > threshold:
                    # Add both directions for undirected graph
                    edges.append([i, j])
                    edges.append([j, i])
        
        # Handle no-edge case
        if not edges and self.fully_connected_fallback:
            return self._create_fully_connected(num_nodes)
        
        # Convert to tensor
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Validate indices
        if torch.any(edge_index >= num_nodes) or torch.any(edge_index < 0):
            min_index = torch.min(edge_index).item()
            max_index = torch.max(edge_index).item()
            print(f"Edge indices out of bounds! Min: {min_index}, Max: {max_index} (should be 0-{num_nodes-1})")
            edge_index = torch.clamp(edge_index, 0, num_nodes-1)
        
        return edge_index

    def _create_fully_connected(self, num_nodes: int) -> torch.LongTensor:
        """Create fully connected graph between TIME STEPS"""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def build_graph_for_batch(self, batch_data: np.ndarray) -> List[torch.LongTensor]:
        """
        Build graphs for a batch of EMG samples.
        
        Args:
            batch_data: EMG time-series of shape (batch_size, time_steps, channels)
            
        Returns:
            edge_indices: List of edge_index tensors for each sample
        """
        if batch_data.ndim != 3:
            raise ValueError(f"Batch input must be 3D (batch, time, channels), got shape {batch_data.shape}")
            
        edge_indices = []
        
        for sample in batch_data:
            edge_index = self.build_graph(sample)
            edge_indices.append(edge_index)
            
        return edge_indices
