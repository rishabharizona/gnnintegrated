import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm

class feat_bottleneck(nn.Module):
    """
    Bottleneck layer for feature transformation
    Options:
    - "ori": No additional processing
    - "bn": Apply batch normalization after linear layer
    """
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        """
        Initialize bottleneck layer
        Args:
            feature_dim: Input feature dimension
            bottleneck_dim: Output feature dimension
            type: Processing type ("ori" or "bn")
        """
        super(feat_bottleneck, self).__init__()
        self.type = type
        
        # Linear transformation layer
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        
        # Batch normalization layer (only used when type="bn")
        if type == "bn":
            self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        
        # Optional components (not currently used in forward pass)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """Forward pass through bottleneck"""
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    """
    Classifier head with optional weight normalization
    Options:
    - "linear": Standard linear classifier
    - "wn": Weight-normalized linear classifier
    """
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        """
        Initialize classifier
        Args:
            class_num: Number of output classes
            bottleneck_dim: Input feature dimension
            type: Classifier type ("linear" or "wn")
        """
        super(feat_classifier, self).__init__()
        self.type = type
        
        # Create appropriate classifier type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        """Forward pass through classifier"""
        return self.fc(x)
