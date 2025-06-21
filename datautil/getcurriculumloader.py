from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import numpy as np
import torch
import math
from collections import defaultdict
import copy

class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        self.dataset.set_labels_by_index(labels, indices, key)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    Stable curriculum learning with gradual exposure and importance weighting
    """
    # Group validation indices by domain
    domain_indices = defaultdict(list)
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        domain = item[2]  # Domain index
        domain_indices[domain].append(idx)

    domain_accuracies = {}
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
            domain_accuracies[domain] = accuracy

    # Get all domains sorted by accuracy (easiest first)
    sorted_domains = sorted(domain_accuracies.keys(), 
                           key=lambda d: domain_accuracies[d], 
                           reverse=True)
    
    print("\n--- Domain Ease Ranking (easiest to hardest) ---")
    for rank, domain in enumerate(sorted_domains, 1):
        print(f"{rank}. Domain {domain}: Accuracy = {domain_accuracies[domain]:.4f}")

    # Conservative curriculum progression
    num_domains = len(sorted_domains)
    
    # Calculate domain inclusion based on training progress
    progress = min(1.0, stage / max(1, args.CL_PHASE_EPOCHS - 1))
    
    # Always include the easiest domain
    min_domains = 1
    
    # Gradually include more domains
    if stage < 3:  # First 3 stages: easiest domain only
        selected_domains = sorted_domains[:1]
    elif stage < 7:  # Next 4 stages: top 25% easiest domains
        selected_domains = sorted_domains[:max(min_domains, int(num_domains * 0.25))]
    elif stage < 12:  # Next 5 stages: top 50% easiest domains
        selected_domains = sorted_domains[:max(min_domains, int(num_domains * 0.5))]
    else:  # Full dataset
        selected_domains = sorted_domains
    
    print(f"Stage {stage}: Selecting {len(selected_domains)} domains")

    # Gather training indices from selected domains
    train_domain_indices = defaultdict(list)
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        domain = item[2]
        train_domain_indices[domain].append(idx)

    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            # Use ALL samples from selected domains
            selected_indices.extend(train_domain_indices[domain])

    print(f"Using {len(selected_indices)} samples from {len(selected_domains)} domains")
    
    # Create curriculum subset
    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    
    # Create a weighted sampler to balance classes
    class_counts = {}
    for idx in selected_indices:
        item = train_dataset[idx]
        label = item[1]
        class_counts[label] = class_counts.get(label, 0) + 1
        
    # Skip weighting if we have too few classes or samples
    if len(class_counts) < 2 or len(selected_indices) < 100:
        curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.N_WORKERS)
    else:
        weights = [1.0 / class_counts[train_dataset[idx][1]] for idx in selected_indices]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                      sampler=sampler, num_workers=args.N_WORKERS)

    return curriculum_loader

def split_dataset_by_domain(dataset, val_ratio=0.2, seed=42):
    """Split dataset while preserving domain distributions"""
    domain_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        domain = item[2]  # Domain index
        domain_indices[domain].append(idx)

    train_indices, val_indices = [], []
    for domain, indices in domain_indices.items():
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, shuffle=True
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
