import torch.nn as nn

# Configuration for different sensor modalities
var_size = {
    'emg': {
        'in_size': 8,        # Number of input channels (EMG sensors)
        'ker_size': 9,       # Kernel size for temporal convolution
        'fc_size': 32 * 44   # Output size before fully-connected layers
    }
}

class ActNetwork(nn.Module):
    """
    Convolutional neural network for sensor-based activity recognition
    Specifically designed for EMG data processing
    
    Architecture:
    - Two convolutional blocks with batch norm and ReLU
    - Each block followed by max pooling
    - Output flattened for feature extraction
    
    Input shape: (batch, channels, 1, time_steps)
    Output shape: (batch, 1408)  # For EMG: 32 * 44 = 1408
    """
    def __init__(self, taskname='emg'):
        """
        Initialize the network
        Args:
            taskname: Sensor modality (currently only 'emg' supported)
        """
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=var_size[taskname]['in_size'],
                out_channels=16,
                kernel_size=(1, var_size[taskname]['ker_size']),
                padding=(0, var_size[taskname]['ker_size']//2)  # Add padding to maintain temporal dimension
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(1, var_size[taskname]['ker_size']),
                padding=(0, var_size[taskname]['ker_size']//2)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        
        # Precomputed output size after convolutions
        self.in_features = var_size[taskname]['fc_size']

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch, channels, 1, time_steps)
        Returns:
            Flattened feature tensor of shape (batch, features)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.in_features)  # Flatten while preserving batch dimension
        return x
