import torch
import torch.nn as nn
import torch.nn.functional as F
from alg.domain_align import DomainAlignLayer

class Algorithm(nn.Module):
    """
    Enhanced base class for all domain generalization algorithms.
    
    Provides common functionality for training/evaluation mode switching,
    domain alignment, and defines the required interface methods.
    """
    def __init__(self, args):
        super(Algorithm, self).__init__()
        self.args = args
        self.train()  # Start in training mode by default
        
        # Domain alignment components
        self.domain_align_weight = getattr(args, 'domain_align_weight', 0.5)
        if self.domain_align_weight > 0:
            self.domain_align = DomainAlignLayer(args.bottleneck_dim)
            print(f"Initialized domain alignment (weight: {self.domain_align_weight})")
    
    def update(self, minibatches):
        """Update the model parameters using a batch of data"""
        raise NotImplementedError("Subclasses must implement update method")

    def predict(self, x):
        """Make predictions on input data"""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def train(self, mode=True):
        """Set the model to training mode"""
        return super().train(mode)
    
    def eval(self):
        """Set the model to evaluation mode"""
        return super().eval()
    
    def explain(self, x):
        """Explain the model's prediction for input x"""
        # Default implementation just returns predictions
        return self.predict(x)
    
    def compute_domain_alignment(self, source_features, target_features):
        """Compute domain alignment loss if enabled"""
        if hasattr(self, 'domain_align') and self.domain_align_weight > 0:
            return self.domain_align_weight * self.domain_align(source_features, target_features)
        return 0.0
    
    def update_bn_stats(self, target_loader):
        """Update batch normalization statistics with target data"""
        original_mode = self.training
        self.eval()
        
        with torch.no_grad():
            for batch in target_loader:
                inputs = batch[0].to(self.args.device)
                if self.args.use_gnn and GNN_AVAILABLE:
                    inputs = transform_for_gnn(inputs)
                _ = self.predict(inputs)  # Forward pass updates BN stats
        
        self.train(original_mode)


class DomainAlignLayer(nn.Module):
    """
    Domain alignment layer for feature distribution matching.
    Combines adversarial training with covariance alignment.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, source, target):
        # Domain adversarial loss
        combined = torch.cat([source, target], dim=0)
        domains = torch.cat([
            torch.zeros(source.size(0)), 
            torch.ones(target.size(0))
        ], dim=0).to(source.device)
        
        domain_pred = self.domain_classifier(combined)
        adv_loss = F.binary_cross_entropy_with_logits(
            domain_pred.squeeze(), 
            domains
        )
        
        # Feature covariance alignment
        source_cov = self.covariance(source)
        target_cov = self.covariance(target)
        cov_loss = F.mse_loss(source_cov, target_cov)
        
        return adv_loss + 0.5 * cov_loss
    
    def covariance(self, x):
        """Compute feature covariance matrix"""
        batch_size = x.size(0)
        x_mean = x - x.mean(dim=0)
        return torch.matmul(x_mean.t(), x_mean) / (batch_size - 1)
