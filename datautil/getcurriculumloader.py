from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import numpy as np
import torch
from collections import defaultdict

class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        self.dataset.set_labels_by_index(labels, indices, key)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    Accuracy-preserving curriculum learning with sample weighting
    """
    # Skip curriculum in later stages
    if stage >= args.CL_PHASE_EPOCHS - 49:
        return DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.N_WORKERS)

    # Group validation indices by domain
    domain_indices = defaultdict(list)
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        domain = item[2]
        domain_indices[domain].append(idx)

    domain_metrics = {}
    algorithm.eval()
    with torch.no_grad():
        for domain, indices in domain_indices.items():
            subset = Subset(val_dataset, indices)
            loader = DataLoader(subset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=args.N_WORKERS)

            correct = 0
            total = 0

            for batch in loader:
                inputs = batch[0].cuda().float()
                labels = batch[1].cuda().long()
                
                output = algorithm.predict(inputs)
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total if total > 0 else 0
            domain_metrics[domain] = accuracy

    # Calculate domain weights (higher weight = easier domain)
    min_acc = min(domain_metrics.values()) if domain_metrics else 0.01
    domain_weights = {d: max(acc, min_acc) for d, acc in domain_metrics.items()}
    total_weight = sum(domain_weights.values())
    domain_weights = {d: w/total_weight for d, w in domain_weights.items()}
    
    # Create sample weights based on domain and class
    sample_weights = []
    class_counts = defaultdict(int)
    
    # First pass: count classes
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        class_counts[item[1]] += 1
        
    # Second pass: calculate weights
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        domain = item[2]
        cls = item[1]
        
        # Domain weight component
        domain_w = domain_weights.get(domain, min_acc/total_weight)
        
        # Class weight component (inverse frequency)
        class_w = 1.0 / max(class_counts[cls], 1)
        
        # Curriculum weighting (focus on easier domains early)
        curriculum_factor = max(0.5, 1.0 - (stage / args.CL_PHASE_EPOCHS))
        weight = (curriculum_factor * domain_w) + ((1 - curriculum_factor) * class_w)
        
        sample_weights.append(weight)

    # Normalize weights
    max_w = max(sample_weights)
    sample_weights = [w / max_w for w in sample_weights]
    
    # Create curriculum sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), 
                                   replacement=False)
    
    # Create standard DataLoader with curriculum sampling
    curriculum_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.N_WORKERS)
    
    print(f"Curriculum stage {stage}: Using weighted sampling across all domains")
    return curriculum_loader

def split_dataset_by_domain(dataset, val_ratio=0.2, seed=42):
    """Split dataset while preserving domain distributions"""
    domain_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        domain = item[2]
        domain_indices[domain].append(idx)

    train_indices, val_indices = [], []
    for domain, indices in domain_indices.items():
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, shuffle=True
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
