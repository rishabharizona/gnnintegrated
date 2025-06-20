import torch
from network import act_network

def get_fea(args):
    """Initialize feature extractor network"""
    return act_network.ActNetwork(args.dataset)

def accuracy(network, loader, weights=None, usedpredict='p'):
    """
    Calculate accuracy for a given data loader
    Args:
        network: Model to evaluate
        loader: Data loader
        weights: Sample weights (optional)
        usedpredict: Which prediction method to use ('p' for predict, otherwise predict1)
    Returns:
        Accuracy score (float)
    """
    # Handle case where loader is None
    if loader is None:
        return 0.0
    
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            
            # Select prediction method
            if usedpredict == 'p':
                p = network.predict(x)
            else:
                p = network.predict1(x)
            
            # Handle sample weights
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:weights_offset + len(x)]
                weights_offset += len(x)
            
            batch_weights = batch_weights.cuda()
            
            # Calculate correct predictions
            if p.size(1) == 1:  # Binary classification
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:  # Multiclass classification
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            
            total += batch_weights.sum().item()
    
    network.train()
    return correct / total if total > 0 else 0.0

def predict_proba(network, x):
    """
    Predict class probabilities
    Args:
        network: Model to use for prediction
        x: Input tensor
    Returns:
        Class probabilities tensor
    """
    network.eval()
    with torch.no_grad():
        x = x.cuda().float()
        logits = network.predict(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
    network.train()
    return probs
