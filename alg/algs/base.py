import torch
import torch.nn as nn
import torch.nn.functional as F
from alg.utils import MovingAverageEMA, apply_spectral_norm
from alg.domain_align import DomainAlignLayer
class Algorithm(nn.Module):
    """
    Enhanced base class for domain generalization algorithms
    with state-of-the-art improvements
    """
    def __init__(self, args):
        super(Algorithm, self).__init__()
        self.args = args
        self.train()
        
        # Domain alignment
        self.domain_align_weight = getattr(args, 'domain_align_weight', 0.7)
        if self.domain_align_weight > 0:
            bottleneck_dim = int(args.bottleneck)
            self.domain_align = DomainAlignLayer(bottleneck_dim)
            print(f"Domain alignment enabled (weight: {self.domain_align_weight})")
        
        # EMA model
        self.ema_decay = getattr(args, 'ema_decay', 0.999)
        if self.ema_decay > 0:
            self.ema = MovingAverageEMA(self, decay=self.ema_decay)
            print(f"EMA enabled (decay: {self.ema_decay})")
        
        # Spectral normalization
        if getattr(args, 'spectral_norm', True):
            apply_spectral_norm(self)
            print("Applied spectral normalization")
    
    def update(self, minibatches):
        """Update model parameters"""
        raise NotImplementedError
    
    def predict(self, x):
        """Make predictions"""
        raise NotImplementedError
    
    def forward(self, x):
        """Forward pass"""
        return self.predict(x)
    
    def ema_update(self, decay=None):
        """Update EMA model"""
        if not hasattr(self, 'ema'):
            return
        self.ema.update(decay or self.ema_decay)
    
    def apply_ema(self):
        """Apply EMA parameters"""
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()
    
    def restore_ema(self):
        """Restore original parameters"""
        if hasattr(self, 'ema'):
            self.ema.restore()
    
    def compute_domain_alignment(self, source_features, target_features):
        """Compute domain alignment loss"""
        if hasattr(self, 'domain_align') and self.domain_align_weight > 0:
            return self.domain_align_weight * self.domain_align(source_features, target_features)
        return 0.0
    
    def update_bn_stats(self, target_loader):
        """Update BN stats with target data"""
        original_mode = self.training
        self.eval()
        
        with torch.no_grad():
            for batch in target_loader:
                inputs = batch[0].to(self.args.device)
                if getattr(self.args, 'use_gnn', False):
                    from train import transform_for_gnn, GNN_AVAILABLE
                    if GNN_AVAILABLE:
                        inputs = transform_for_gnn(inputs)
                _ = self.predict(inputs)
        
        self.train(original_mode)
    
    def apply_spectral_norm(self):
        """Apply spectral normalization"""
        apply_spectral_norm(self)

class DomainAlignLayer(nn.Module):
    """Advanced domain alignment with multi-kernel MMD"""
    def __init__(self, feature_dim):
        super().__init__()
        self.scales = [1, 2, 4]  # Multiple kernel scales
    
    def forward(self, source, target):
        mmd_loss = 0
        for scale in self.scales:
            mmd_loss += self.mmd_rbf(source, target, scale)
        return mmd_loss / len(self.scales)
    
    def mmd_rbf(self, x, y, sigma):
        """Gaussian kernel MMD"""
        xx = self.gaussian_kernel(x, x, sigma)
        yy = self.gaussian_kernel(y, y, sigma)
        xy = self.gaussian_kernel(x, y, sigma)
        return xx.mean() + yy.mean() - 2 * xy.mean()
    
    def gaussian_kernel(self, a, b, sigma):
        """Compute Gaussian kernel matrix"""
        dist = torch.cdist(a, b, p=2)
        return torch.exp(-dist**2 / (2 * sigma**2))
