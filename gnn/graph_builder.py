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

    def build_graph(self, data_sample: np.ndarray) -> torch.LongTensor:
        """
        Build graph edge indices from EMG sample for batch processing.
        
        Args:
            data_sample: EMG time-series of shape (time_steps, channels)
            
        Returns:
            edge_index: Tensor of shape [2, num_edges]
        """
        # Validate input
        if not isinstance(data_sample, np.ndarray):
            raise TypeError(f"Input must be numpy array, got {type(data_sample)}")
            
        if data_sample.ndim != 2:
            raise ValueError(f"Input must be 2D (time_steps, channels), got shape {data_sample.shape}")
            
        T, C = data_sample.shape
        if T < 2:
            raise ValueError(f"Need at least 2 time steps, got {T}")
        if C < 2:
            raise ValueError(f"Need at least 2 channels, got {C}")

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity(data_sample)
        
        # Determine threshold
        threshold = self._determine_threshold(similarity_matrix)
        
        # Build edges
        return self._create_edges(similarity_matrix, threshold, C)

    def _compute_similarity(self, data: np.ndarray) -> np.ndarray:
        """Compute similarity matrix based on selected method"""
        if self.method == 'correlation':
            # Add small epsilon to avoid division by zero
            data = data + 1e-8
            return np.corrcoef(data.T)
        elif self.method == 'covariance':
            return np.cov(data.T)
        elif self.method == 'euclidean':
            dist_matrix = squareform(pdist(data.T, 'euclidean'))
            # Convert distance to similarity (inverse relationship)
            max_dist = np.max(dist_matrix)
            return 1 - (dist_matrix / max_dist) if max_dist > 0 else np.ones_like(dist_matrix)

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

    def _create_edges(self, matrix: np.ndarray, threshold: float, num_channels: int) -> torch.LongTensor:
        """Create edge connections without weights"""
        edges = []
        
        # Create edges where absolute similarity exceeds threshold
        for i, j in itertools.combinations(range(num_channels), 2):
            similarity = matrix[i, j]
            if abs(similarity) > threshold:
                edges.append([i, j])
                edges.append([j, i])  # Undirected graph
                
        # Handle no-edge case
        if not edges and self.fully_connected_fallback:
            return self._create_fully_connected(num_channels)
        
        # Convert to tensor
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index

    def _create_fully_connected(self, num_channels: int) -> torch.LongTensor:
        """Create fully connected graph"""
        edges = []
        for i in range(num_channels):
            for j in range(num_channels):
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
