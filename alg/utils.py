import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAverageEMA:
    """Exponential Moving Average for model stability"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, features, labels):
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T)
        
        # Mask for positive samples (same class)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # Negative mask
        negative_mask = 1 - mask
        
        # Positive similarity
        positives = torch.sum(similarity_matrix * mask, dim=1)
        positive_count = mask.sum(dim=1)
        positive_loss = -torch.log(positives / (positives + negative_mask.sum(dim=1)))
        
        return positive_loss.mean()

def apply_spectral_norm(model):
    """Apply spectral normalization to all convolutional layers"""
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.utils.spectral_norm(module)
        else:
            apply_spectral_norm(module)  # Recursive
