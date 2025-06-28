import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from gnn.graph_builder import GraphBuilder
import numpy as np

class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network for sensor-based activity recognition
    Combines 1D convolutions with dynamic graph construction
    """
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.graph_builder = graph_builder or GraphBuilder()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_features = input_dim
        self.out_features = output_dim
        
        # Temporal feature extractor
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Graph convolutions
        self.gcn1 = GCNConv(32, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Classifier with residual connection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Reconstruction decoder (optional)
        self.recon_decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        ) if output_dim != input_dim else nn.Identity()

    def forward(self, x, return_features=False):
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.squeeze(2)
        
        # Ensure correct shape: [batch, channels, timesteps]
        if x.size(1) != self.input_dim:
            x = x.permute(0, 2, 1)
        
        batch_size, channels, timesteps = x.shape
        
        # Temporal convolution
        x_conv = self.temporal_conv(x)
        _, features, reduced_timesteps = x_conv.shape
        
        # Prepare for GCN: [batch, time, features]
        x_gcn = x_conv.permute(0, 2, 1)
        
        # Build graph from current batch features
        with torch.no_grad():
            # Create signature for batch representation
            batch_signature = torch.mean(x_gcn, dim=[0, 2]).item()
            edge_index = self.graph_builder.build_graph(x_gcn)
            
            # Validate graph indices
            max_index = batch_size * reduced_timesteps - 1
            if torch.any(edge_index > max_index):
                edge_index = torch.clamp(edge_index, 0, max_index)

        # Batch the graph with node offsetting
        edge_indices = []
        for i in range(batch_size):
            offset = i * reduced_timesteps
            edge_index_offset = edge_index + offset
            edge_indices.append(edge_index_offset)
        edge_index = torch.cat(edge_indices, dim=1).to(x.device)

        # Process through GCN
        x_flat = x_gcn.reshape(batch_size * reduced_timesteps, -1)
        x_gcn1 = F.relu(self.gcn1(x_flat, edge_index))
        x_gcn2 = F.relu(self.gcn2(x_gcn1, edge_index))
        
        # Reshape and pool
        x_pooled = x_gcn2.reshape(batch_size, reduced_timesteps, -1)
        x_pooled = torch.mean(x_pooled, dim=1)
        
        # Classification
        logits = self.fc(x_pooled)
        
        if return_features:
            return logits, x_pooled
        return logits

    def reconstruct(self, features):
        """Reconstruct input features from embeddings"""
        return self.recon_decoder(features)
