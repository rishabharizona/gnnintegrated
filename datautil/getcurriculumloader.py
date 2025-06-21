from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
import random

class SubsetWithLabelSetter(Subset):
    """Custom Subset class that supports label-setting methods"""
    def set_labels_by_index(self, labels, indices, key):
        """Delegate label-setting to the underlying dataset"""
        # Map subset indices to original dataset indices
        original_indices = [self.indices[i] for i in indices]
        self.dataset.set_labels_by_index(labels, original_indices, key)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    Enhanced curriculum learning with domain difficulty scoring and adaptive sampling
    """
    # Group validation indices by domain
    domain_indices = {}
    for idx in range(len(val_dataset)):
        domain = val_dataset[idx][2]  # Extract domain from index 2
        domain_indices.setdefault(domain, []).append(idx)

    domain_metrics = []

    # Compute loss and accuracy for each domain
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

            for batch in loader:
                inputs = batch[0].cuda().float()
                labels = batch[1].cuda().long()
                
                output = algorithm.predict(inputs)
                loss = torch.nn.functional.cross_entropy(output, labels)
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            accuracy = correct / total if total > 0 else 0
            domain_metrics.append((domain, avg_loss, accuracy))

    # Calculate domain difficulty scores
    losses = [m[1] for m in domain_metrics]
    min_loss, max_loss = min(losses), max(losses)
    accs = [m[2] for m in domain_metrics]
    min_acc, max_acc = min(accs), max(accs)
    
    domain_scores = []
    for domain, loss, acc in domain_metrics:
        # Normalize metrics (0-1 range)
        norm_loss = (loss - min_loss) / (max_loss - min_loss + 1e-10)
        norm_acc = (acc - min_acc) / (max_acc - min_acc + 1e-10)
        
        # Difficulty score: higher = harder (70% loss, 30% inverse accuracy)
        difficulty = 0.7 * norm_loss + 0.3 * (1 - norm_acc)
        domain_scores.append((domain, difficulty))

    # Sort by easiest domains first
    domain_scores.sort(key=lambda x: x[1])
    
    # Print domain ranking
    print("\n--- Domain Difficulty Ranking (easiest to hardest) ---")
    for rank, (domain, difficulty) in enumerate(domain_scores, 1):
        print(f"{rank}. Domain {domain}: Difficulty = {difficulty:.4f}")

    # Curriculum progression with adaptive pacing
    num_domains = len(domain_scores)
    progress = min(1.0, (stage + 1) / args.CL_PHASE_EPOCHS)
    progress = np.sqrt(progress)  # Slower initial progression
    
    # Determine number of domains to include
    num_selected = max(2, min(num_domains, int(np.ceil(progress * num_domains * 0.8)))
    selected_domains = [domain for domain, _ in domain_scores[:num_selected]]
    
    # Add random harder domain for diversity
    if len(domain_scores) > num_selected:
        random_hard_domain = random.choice(domain_scores[num_selected:])[0]
        selected_domains.append(random_hard_domain)
        print(f"Adding random harder domain: {random_hard_domain}")

    # Gather training indices from selected domains
    train_domain_indices = {}
    for idx in range(len(train_dataset)):
        domain = train_dataset[idx][2]
        train_domain_indices.setdefault(domain, []).append(idx)

    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_indices = train_domain_indices[domain]
            # Balanced sampling: Use up to 80% of domain samples
            n_samples = min(len(domain_indices), max(50, int(len(domain_indices) * 0.8)))
            selected_indices.extend(random.sample(domain_indices, n_samples))

    print(f"Selected {len(selected_indices)} samples from {len(selected_domains)} domains")
    
    # Create curriculum subset with label-setting capability
    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS,
                                   drop_last=True)

    return curriculum_loader

def split_dataset_by_domain(dataset, val_ratio=0.2, seed=42):
    """Split dataset while preserving domain distribution"""
    domain_indices = {}
    for idx in range(len(dataset)):
        domain = dataset[idx][2]  # Extract domain from index 2
        domain_indices.setdefault(domain, []).append(idx)

    train_indices, val_indices = [], []
    for domain, indices in domain_indices.items():
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, shuffle=True
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
