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
    Robust curriculum learning with gradual exposure and distribution preservation
    """
    # Group validation indices by domain
    domain_indices = defaultdict(list)
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        domain = item[2]  # Domain index
        domain_indices[domain].append(idx)

    domain_metrics = []

    # Compute accuracy for each domain
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
            domain_metrics.append((domain, accuracy))

    # Sort by easiest domains first (highest accuracy)
    domain_metrics.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- Domain Ease Ranking (easiest to hardest) ---")
    for rank, (domain, accuracy) in enumerate(domain_metrics, 1):
        print(f"{rank}. Domain {domain}: Accuracy = {accuracy:.4f}")

    # Conservative curriculum progression
    num_domains = len(domain_metrics)
    
    # Slower progression: start with 1 domain, gradually include more
    progress = min(1.0, (stage + 1) / args.CL_PHASE_EPOCHS)
    
    # Start with easiest domain only in first stage
    if stage < 3:
        num_selected = 1
    else:
        # Gradually include more domains
        num_selected = min(num_domains, 1 + int(progress * (num_domains - 1)))
    
    selected_domains = [domain for domain, _ in domain_metrics[:num_selected]]
    print(f"Selecting domains: {selected_domains}")

    # Gather ALL training indices from selected domains
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
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS,
                                   drop_last=False)  # Preserve all samples

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
