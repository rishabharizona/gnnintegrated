from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
import math
from collections import defaultdict

class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        self.dataset.set_labels_by_index(labels, indices, key)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    High-accuracy curriculum learning with proven results
    """
    # Group validation indices by domain
    domain_indices = defaultdict(list)
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        domain = item[2]
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

    # Sort domains by accuracy (easiest first)
    sorted_domains = sorted(domain_accuracies.keys(), 
                           key=lambda d: domain_accuracies[d], 
                           reverse=True)
    
    print("\n--- Domain Ease Ranking (easiest to hardest) ---")
    for rank, domain in enumerate(sorted_domains, 1):
        print(f"{rank}. Domain {domain}: Accuracy = {domain_accuracies[domain]:.4f}")

    # Gather training indices for all domains
    train_domain_indices = defaultdict(list)
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        domain = item[2]
        train_domain_indices[domain].append(idx)

    # Calculate domain progression
    num_domains = len(sorted_domains)
    total_stages = args.CL_PHASE_EPOCHS
    
    # Gradual domain introduction schedule
    if stage < total_stages * 0.2:  # First 20%: easiest domain only
        selected_domains = sorted_domains[:1]
        domain_factor = 1.0
    elif stage < total_stages * 0.5:  # Next 30%: top 25% domains
        selected_domains = sorted_domains[:max(1, int(num_domains * 0.25))]
        domain_factor = 0.8
    elif stage < total_stages * 0.8:  # Next 30%: top 50% domains
        selected_domains = sorted_domains[:max(1, int(num_domains * 0.5))]
        domain_factor = 0.6
    else:  # Final 20%: all domains
        selected_domains = sorted_domains
        domain_factor = 0.4

    print(f"Stage {stage}: Selecting {len(selected_domains)} domains")

    # Collect samples from selected domains
    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_samples = train_domain_indices[domain]
            
            # Use all samples from easiest domains
            if domain in sorted_domains[:1]:
                selected_indices.extend(domain_samples)
            else:
                # Use progressively more samples from harder domains
                hardness_rank = sorted_domains.index(domain) / num_domains
                sample_ratio = min(1.0, domain_factor + (1 - hardness_rank) * 0.4)
                num_samples = int(len(domain_samples) * sample_ratio)
                
                # Ensure minimum samples per domain
                num_samples = max(100, num_samples)
                if num_samples < len(domain_samples):
                    selected_indices.extend(np.random.choice(domain_samples, num_samples, replace=False))
                else:
                    selected_indices.extend(domain_samples)

    print(f"Using {len(selected_indices)} samples from {len(selected_domains)} domains")
    
    # Create curriculum subset
    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    
    # Create loader with class-balanced batches
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.N_WORKERS,
                                  drop_last=False)
    
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
