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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_features = input_dim  # For compatibility with algorithms
        self.out_features = output_dim  # For compatibility with algorithms
        
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
        
        # Reconstruction layer for pretraining
        self.recon = nn.Linear(output_dim, input_dim)  # For mean feature reconstruction

    def forward(self, x):
        # Handle different input dimensions
        if x.dim() == 2:
            # Add timestep dimension: [batch, features] -> [batch, 1, features]
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # Flatten extra dimension: [batch, channels, 1, timesteps] -> [batch, channels, timesteps]
            x = x.squeeze(2)
        
        # Now x should be 3D: [batch, channels, timesteps]
        # Convert to [batch, channels, timesteps] for Conv1d
        if x.size(1) != self.input_dim:
            # Input is [batch, timesteps, channels] -> permute to [batch, channels, timesteps]
            x = x.permute(0, 2, 1)
        
        batch_size, channels, timesteps = x.shape
        
        # Temporal convolution: [batch, features, timesteps]
        x = self.temporal_conv(x)  # Output: [batch, 32, timesteps//4]
        _, features, reduced_timesteps = x.shape
        
        # Prepare for GCN: [batch, features, time] -> [batch, time, features]
        x = x.permute(0, 2, 1)  # [batch, reduced_timesteps, 32]
        x = x.reshape(batch_size * reduced_timesteps, -1)
        
        # Build graph directly in forward pass
        # Get first sample's features for graph building
        sample = x[:reduced_timesteps].detach().cpu().numpy()
        edge_index = self.graph_builder.build_graph(sample)
        
        # Handle empty graph case
        if edge_index.numel() == 0:
            # Use fully connected graph as fallback
            num_nodes = reduced_timesteps
            edge_index = torch.tensor(
                [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                dtype=torch.long
            ).t().contiguous()
        
        # Move to device and clone to avoid warnings
        edge_index = edge_index.to(x.device).clone().detach()
        
        # Batch the graph with node offsetting
        edge_indices = []
        for i in range(batch_size):
            offset = i * reduced_timesteps
            edge_index_offset = edge_index + offset
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
        """Reconstruct mean input features for pretraining"""
        return self.recon(features)
