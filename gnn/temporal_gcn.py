import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TemporalGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph convolutional layers
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Reconstruction layer
        self.reconstruct_layer = nn.Linear(output_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Input shape: (batch_size, seq_len, input_dim)
        
        # First graph convolution
        x = F.relu(self.bn1(self.gc1(x).transpose(1, 2)).transpose(1, 2))
        x = self.dropout(x)
        
        # Second graph convolution
        x = F.relu(self.bn2(self.gc2(x).transpose(1, 2)).transpose(1, 2))
        x = self.dropout(x)
        
        # Global average pooling over temporal dimension
        x = torch.mean(x, dim=1)
        
        return x

    def reconstruct(self, features):
        """
        Reconstruct input from features
        Args:
            features: Feature tensor from forward pass
        Returns:
            Reconstructed input
        """
        return self.reconstruct_layer(features)
