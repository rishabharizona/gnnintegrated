import torch
from network import act_network
from gnn.temporal_gcn import TemporalGCN

def get_fea(args):
    """Initialize feature extractor network with GNN support"""
    if hasattr(args, 'model_type') and args.model_type == 'gnn':
        net = TemporalGCN(
            input_dim=8,  # EMG channels
            hidden_dim=getattr(args, 'gnn_hidden_dim', 64),
            output_dim=getattr(args, 'gnn_output_dim', 256)
        )
        net.in_features = net.output_dim  # Needed for downstream bottleneck
        return net
    return act_network.ActNetwork(args.dataset)

def accuracy(network, loader, transform_fn=None):
    """
    Calculate accuracy with reduced complexity
    - Removed sample weighting to prevent over-optimization
    - Unified prediction method
    - Simplified binary/multiclass handling
    """
    if loader is None:
        return 0.0
    
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()

            if transform_fn:
                x = transform_fn(x)
                
            p = network.predict(x)
            
            # Unified dimension handling
            if p.dim() > 2:
                p = p.squeeze(1)
            
            # Combined binary/multiclass handling
            if p.size(1) == 1:  # Binary classification
                pred = (p > 0).long().squeeze(1)
            else:  # Multiclass classification
                pred = p.argmax(1)
            
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    network.train()
    return correct / total if total > 0 else 0.0

def predict_proba(network, x):
    """Predict class probabilities with simplified output"""
    network.eval()
    with torch.no_grad():
        x = x.cuda().float()
        logits = network.predict(x)
        
        # Handle multi-dimensional outputs
        if logits.dim() > 2:
            logits = logits.squeeze(1)
            
        # Handle binary vs multiclass outputs
        if logits.size(1) == 1:
            # Binary classification: sigmoid + expand to 2 classes
            prob_positive = torch.sigmoid(logits)
            probs = torch.cat([1 - prob_positive, prob_positive], dim=1)
        else:
            probs = torch.nn.functional.softmax(logits, dim=1)
    
    network.train()
    return probs
