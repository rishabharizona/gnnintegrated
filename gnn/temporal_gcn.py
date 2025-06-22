import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network for sensor-based activity recognition
    Combines 1D convolutions for temporal features with GCN for spatial features
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
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
        
    def forward(self, data):
        # Data is either a tuple (x, edge_index) or a PyG Data object
        if isinstance(data, tuple):
            x, edge_index = data
            batch = None
        else:
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            
        # Temporal convolution: [batch, channels, timesteps]
        x = x.permute(0, 2, 1)  # [batch, features, timesteps]
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)  # [batch, timesteps, features]
        
        # Prepare for GCN: flatten batch and timesteps
        batch_size, timesteps, features = x.shape
        x = x.reshape(batch_size * timesteps, features)
        
        # Graph convolution
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        
        # Pool over time
        x = x.reshape(batch_size, timesteps, -1)
        x = torch.mean(x, dim=1)  # Global average pooling over time
        
        # Classification
        return self.fc(x)
