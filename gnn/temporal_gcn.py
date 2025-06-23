import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from gnn.graph_builder import GraphBuilder
import numpy as np

class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network for sensor-based activity recognition
    Combines 1D convolutions for temporal features with GCN for spatial features
    """
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.graph_builder = graph_builder or GraphBuilder()
        # Temporal feature extractor
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Spatial graph convolutions
        self.gcn1 = GCNConv(32, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # Classifier
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Store input/output dimensions for compatibility
        self.in_features = input_dim
        self.out_features = output_dim

    def forward(self, x):
        # Handle different input dimensions
        if x.dim() == 2:
            # Add timestep dimension: [batch, features] -> [batch, 1, features]
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # Flatten extra dimension: [batch, channels, 1, timesteps] -> [batch, channels, timesteps]
            x = x.squeeze(2)
        
        # Now x should be 3D: [batch, channels, timesteps]
        batch_size, channels, timesteps = x.shape
        
        # Temporal convolution: [batch, features, timesteps]
        x = self.temporal_conv(x)  # Output: [batch, 32, timesteps//4]
        _, _, reduced_timesteps = x.shape
        
        # Prepare for GCN: [batch, features, time] -> [batch*time, features]
        x = x.permute(0, 2, 1)  # [batch, reduced_timesteps, 32]
        x = x.reshape(batch_size * reduced_timesteps, -1)
        
        # Build graph directly in forward pass
        # Get first sample's features for graph building
        sample = x[:reduced_timesteps].detach().cpu().numpy()
        edge_index_single = self.graph_builder.build_graph(sample)
        edge_index_single = torch.tensor(edge_index_single, dtype=torch.long).to(x.device)
        
        # Batch the graph with node offsetting
        edge_indices = []
        for i in range(batch_size):
            offset = i * reduced_timesteps
            edge_index_offset = edge_index_single + offset
            edge_indices.append(edge_index_offset)
        edge_index = torch.cat(edge_indices, dim=1)
        
        # Graph convolution
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        
        # Reshape back: [batch, reduced_timesteps, features]
        x = x.reshape(batch_size, reduced_timesteps, -1)
        
        # Global pooling over time
        x = torch.mean(x, dim=1)  # [batch, features]
        return self.fc(x)

    def reconstruct(self, features):
        """Dummy reconstruction method for pretraining"""
        # Simple linear reconstruction
        return torch.matmul(features, self.fc.weight.T)
