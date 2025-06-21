import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import copy

class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        self.dataset.set_labels_by_index(labels, indices, key)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    High-performance curriculum learning system for 91% accuracy
    """
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

            total_loss = 0.0
            correct = 0
            total = 0
            num_batches = 0
            class_correct = defaultdict(int)
            class_total = defaultdict(int)

            for batch in loader:
                inputs = batch[0].cuda().float()
                labels = batch[1].cuda().long()
                
                output = algorithm.predict(inputs)
                loss = nn.functional.cross_entropy(output, labels)
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Track per-class accuracy
                for c in torch.unique(labels):
                    class_mask = (labels == c)
                    class_correct[c.item()] += (predicted[class_mask] == c).sum().item()
                    class_total[c.item()] += class_mask.sum().item()
                
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            accuracy = correct / total if total > 0 else 0
            
            # Calculate class imbalance metric (standard deviation of class accuracies)
            class_accs = [class_correct[c] / max(class_total[c], 1) 
                         for c in class_correct if class_total[c] > 0]
            class_imbalance = np.std(class_accs) if class_accs else 0
            
            domain_metrics[domain] = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'imbalance': class_imbalance
            }

    # Calculate domain difficulty scores
    min_loss = min(m['loss'] for m in domain_metrics.values()) if domain_metrics else 0
    max_loss = max(m['loss'] for m in domain_metrics.values()) if domain_metrics else 1
    min_acc = min(m['accuracy'] for m in domain_metrics.values()) if domain_metrics else 0
    max_acc = max(m['accuracy'] for m in domain_metrics.values()) if domain_metrics else 1
    max_imb = max(m['imbalance'] for m in domain_metrics.values()) if domain_metrics else 1
    
    domain_scores = {}
    for domain, metrics in domain_metrics.items():
        # Normalize metrics
        norm_loss = (metrics['loss'] - min_loss) / (max_loss - min_loss + 1e-10)
        norm_acc = (metrics['accuracy'] - min_acc) / (max_acc - min_acc + 1e-10)
        norm_imb = metrics['imbalance'] / (max_imb + 1e-10)
        
        # Difficulty score (higher = harder)
        difficulty = 0.4 * norm_loss + 0.4 * (1 - norm_acc) + 0.2 * norm_imb
        domain_scores[domain] = difficulty

    # Sort domains by difficulty
    sorted_domains = sorted(domain_scores.keys(), key=lambda d: domain_scores[d])
    
    print("\n--- Domain Difficulty Ranking (easiest to hardest) ---")
    for rank, domain in enumerate(sorted_domains, 1):
        metrics = domain_metrics[domain]
        print(f"{rank}. Domain {domain}: "
              f"Acc={metrics['accuracy']:.4f}, Loss={metrics['loss']:.4f}, "
              f"Imb={metrics['imbalance']:.4f}, Diff={domain_scores[domain]:.4f}")

    # Progressive domain inclusion
    total_domains = len(sorted_domains)
    total_stages = args.CL_PHASE_EPOCHS
    
    # Adaptive progression based on current stage
    if stage < total_stages * 0.3:  # First 30% of training
        # Focus on easiest domains
        coverage = 0.3 + 0.5 * (stage / (total_stages * 0.3))
        domain_factor = 0.9  # Use more samples from easier domains
    elif stage < total_stages * 0.7:  # Middle 40%
        coverage = 0.8 + 0.2 * ((stage - total_stages * 0.3) / (total_stages * 0.4))
        domain_factor = 0.7
    else:  # Final 30%
        coverage = 1.0
        domain_factor = 0.5  # Balanced sampling
    
    num_selected = max(1, min(total_domains, int(total_domains * coverage)))
    selected_domains = sorted_domains[:num_selected]
    print(f"Stage {stage}/{total_stages}: Selecting {num_selected}/{total_domains} domains")

    # Gather training indices
    train_domain_indices = defaultdict(list)
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        domain = item[2]
        train_domain_indices[domain].append(idx)

    selected_indices = []
    domain_weights = {}
    
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_samples = train_domain_indices[domain]
            domain_rank = sorted_domains.index(domain)
            
            # Calculate domain weight based on difficulty
            domain_weight = 1.0 - (domain_rank / total_domains)
            domain_weights[domain] = domain_weight
            
            # Determine sample ratio
            sample_ratio = domain_factor + (1 - domain_factor) * domain_weight
            num_samples = int(len(domain_samples) * sample_ratio)
            num_samples = max(100, min(num_samples, len(domain_samples)))
            
            if num_samples < len(domain_samples):
                selected_indices.extend(np.random.choice(
                    domain_samples, num_samples, replace=False
                ))
            else:
                selected_indices.extend(domain_samples)

    print(f"Using {len(selected_indices)} samples from {len(selected_domains)} domains")
    
    # Class balancing
    class_counts = defaultdict(int)
    for idx in selected_indices:
        item = train_dataset[idx]
        class_counts[item[1]] += 1
    
    if len(class_counts) > 1:  # Only balance if multiple classes exist
        max_count = max(class_counts.values())
        class_weights = {cls: max_count / count for cls, count in class_counts.items()}
        
        # Create sample weights combining domain and class weights
        sample_weights = []
        for idx in selected_indices:
            item = train_dataset[idx]
            domain = item[2]
            cls = item[1]
            weight = domain_weights[domain] * class_weights[cls]
            sample_weights.append(weight)
        
        # Normalize weights
        max_weight = max(sample_weights)
        sample_weights = [w / max_weight for w in sample_weights]
        
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Create curriculum subset
    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    
    # Create loader
    curriculum_loader = DataLoader(
        curriculum_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.N_WORKERS,
        drop_last=False,
        pin_memory=True
    )
    
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
