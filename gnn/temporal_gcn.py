import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TemporalGCN(nn.Module):
    """Simplified Temporal GCN for sensor-based activity recognition"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph convolutions
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        
        # Classifier
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        """Simplified forward pass"""
        # Handle input dimensions
        if x.dim() == 4:  # [batch, channels, 1, time]
            x = x.squeeze(2).permute(0, 2, 1)
        elif x.dim() == 3:  # [batch, time, features]
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        batch_size, timesteps, features = x.shape
        
        # Reshape for GCN: [batch*timesteps, features]
        x_flat = x.reshape(batch_size * timesteps, features)
        
        # Create fully connected graph
        edge_index = []
        for i in range(timesteps):
            for j in range(timesteps):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = edge_index.repeat(1, batch_size).to(x.device)
        
        # Process through GCN
        x_gcn = F.relu(self.gcn1(x_flat, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        
        # Reshape and pool
        x_pooled = x_gcn.reshape(batch_size, timesteps, -1)
        x_pooled = torch.mean(x_pooled, dim=1)
        
        # Classification
        return self.fc(x_pooled)
