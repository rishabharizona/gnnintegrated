import numpy as np
import torch
from torch_geometric.data import Data

def convert_to_graph(sensor_data, adjacency_strategy='fully_connected', threshold=0.5):
    """
    Convert sensor data to graph representation for GNN models
    Args:
        sensor_data: Tensor of shape (num_sensors, timesteps, features)
        adjacency_strategy: Graph construction method ('fully_connected', 'correlation', 'knn')
        threshold: Correlation threshold for 'correlation' strategy
    Returns:
        PyG Data object with node features, edge indices, and edge attributes
    """
    num_nodes = sensor_data.shape[0]
    timesteps = sensor_data.shape[1]
    num_features = sensor_data.shape[2]
    
    # Node features: flatten time series
    x = sensor_data.reshape(num_nodes, -1)  # Shape: [num_nodes, timesteps*features]
    
    # Edge construction
    if adjacency_strategy == 'fully_connected':
        # Create edges between all node pairs (except self)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = None
        
    elif adjacency_strategy == 'correlation':
        # Compute correlation matrix between sensors
        flat_data = x.cpu().numpy()
        corr_matrix = np.corrcoef(flat_data)
        
        # Create edges based on correlation threshold
        edge_index = []
        edge_weight = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if abs(corr_matrix[i, j]) > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Undirected graph
                    weight = abs(corr_matrix[i, j])
                    edge_weight.extend([weight, weight])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1) if edge_weight else None
        
    elif adjacency_strategy == 'knn':
        # K-nearest neighbors based on Euclidean distance
        from sklearn.neighbors import kneighbors_graph
        knn_graph = kneighbors_graph(flat_data, n_neighbors=3, mode='connectivity')
        edge_index = []
        rows, cols = knn_graph.nonzero()
        for i, j in zip(rows, cols):
            if i != j:  # Avoid self-loops
                edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = None
        
    else:
        raise ValueError(f"Unknown adjacency strategy: {adjacency_strategy}")
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
