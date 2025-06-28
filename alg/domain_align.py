import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAlignLayer(nn.Module):
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
        loss = F.binary_cross_entropy_with_logits(
            domain_pred.squeeze(), 
            domains
        )
        
        # Feature covariance alignment
        source_cov = self.covariance(source)
        target_cov = self.covariance(target)
        cov_loss = F.mse_loss(source_cov, target_cov)
        
        return loss + 0.5 * cov_loss
    
    def covariance(self, x):
        batch_size = x.size(0)
        x_mean = x - x.mean(dim=0)
        return torch.matmul(x_mean.t(), x_mean) / (batch_size - 1)
