from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
import random
import math

class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        self.dataset.set_labels_by_index(labels, indices, key)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    Enhanced curriculum learning with dynamic domain selection and balanced sampling
    """
    # Group validation indices by domain
    domain_indices = {}
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]  # Get full item
        domain = item[2]  # Domain is at index 2
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
            class_correct = {}
            class_total = {}

            for batch in loader:
                inputs = batch[0].cuda().float()
                labels = batch[1].cuda().long()
                
                output = algorithm.predict(inputs)
                loss = torch.nn.functional.cross_entropy(output, labels)
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Track per-class accuracy
                for c in range(output.size(1)):
                    class_mask = (labels == c)
                    class_correct[c] = class_correct.get(c, 0) + (predicted[class_mask] == c).sum().item()
                    class_total[c] = class_total.get(c, 0) + class_mask.sum().item()
                
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            accuracy = correct / total if total > 0 else 0
            
            # Calculate class imbalance metric
            class_accs = [class_correct.get(c, 0) / max(class_total.get(c, 1), 1) for c in range(output.size(1))]
            class_imbalance = np.std(class_accs) if class_accs else 0
            
            domain_metrics.append((domain, avg_loss, accuracy, class_imbalance))

    # Calculate domain difficulty scores
    losses = [m[1] for m in domain_metrics]
    min_loss, max_loss = min(losses), max(losses)
    accs = [m[2] for m in domain_metrics]
    min_acc, max_acc = min(accs), max(accs)
    imbalances = [m[3] for m in domain_metrics]
    max_imbalance = max(imbalances) if imbalances else 1
    
    domain_scores = []
    for domain, loss, acc, imb in domain_metrics:
        # Normalize metrics
        norm_loss = (loss - min_loss) / (max_loss - min_loss + 1e-10)
        norm_acc = (acc - min_acc) / (max_acc - min_acc + 1e-10)
        norm_imb = imb / (max_imbalance + 1e-10)
        
        # Difficulty score
        difficulty = 0.5 * norm_loss + 0.3 * (1 - norm_acc) + 0.2 * norm_imb
        domain_scores.append((domain, difficulty))

    # Sort by easiest domains first
    domain_scores.sort(key=lambda x: x[1])
    
    print("\n--- Domain Difficulty Ranking (easiest to hardest) ---")
    for rank, (domain, difficulty) in enumerate(domain_scores, 1):
        print(f"{rank}. Domain {domain}: Difficulty = {difficulty:.4f}")

    # Dynamic curriculum progression
    num_domains = len(domain_scores)
    
    # Cosine annealing schedule
    progress = min(1.0, (stage + 1) / args.CL_PHASE_EPOCHS)
    cos_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
    
    # Base selection
    base_selection = max(1, int(np.ceil(np.log(stage + 2) / np.log(2) * num_domains / 3)))
    
    # Final selection
    num_selected = min(num_domains, max(2, base_selection + int(cos_progress * num_domains * 0.5)))
    
    selected_domains = [domain for domain, _ in domain_scores[:num_selected]]
    
    # Include hardest domain earlier for regularization
    if progress > 0.3 and num_domains > num_selected:
        hardest_domain = domain_scores[-1][0]
        if hardest_domain not in selected_domains:
            selected_domains.append(hardest_domain)
            print(f"Adding hardest domain for regularization: {hardest_domain}")

    # Gather training indices from selected domains
    train_domain_indices = {}
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]  # Get full item
        domain = item[2]  # Domain is at index 2
        train_domain_indices.setdefault(domain, []).append(idx)

    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_indices = train_domain_indices[domain]
            
            # Dynamic sampling
            if progress < 0.5:
                n_samples = min(len(domain_indices), max(100, int(len(domain_indices) * 0.7)))
            else:
                domain_rank = [d for d, _ in domain_scores].index(domain)
                hardness_factor = 0.5 + (domain_rank / num_domains) * 0.5
                n_samples = min(len(domain_indices), max(100, int(len(domain_indices) * hardness_factor)))
            
            selected_indices.extend(random.sample(domain_indices, n_samples))

    print(f"Selected {len(selected_indices)} samples from {len(selected_domains)} domains")
    
    # Class-balanced sampling
    if progress > 0.2 and len(selected_indices) > 0:
        class_indices = {}
        for idx in selected_indices:
            item = train_dataset[idx]  # Get full item
            label = item[1]  # Label is at index 1
            class_indices.setdefault(label, []).append(idx)
        
        # Determine minimum class count
        min_class_count = min(len(indices) for indices in class_indices.values()) if class_indices else 0
        min_class_count = max(50, min_class_count)  # Set minimum per class
        
        balanced_indices = []
        for indices in class_indices.values():
            if len(indices) > min_class_count:
                balanced_indices.extend(random.sample(indices, min_class_count))
            else:
                balanced_indices.extend(indices)
        
        selected_indices = balanced_indices
        print(f"After class balancing: {len(selected_indices)} samples")

    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS,
                                   drop_last=True)

    return curriculum_loader

def split_dataset_by_domain(dataset, val_ratio=0.2, seed=42):
    domain_indices = {}
    for idx in range(len(dataset)):
        item = dataset[idx]  # Get full item
        domain = item[2]  # Domain is at index 2
        domain_indices.setdefault(domain, []).append(idx)

    train_indices, val_indices = [], []
    for domain, indices in domain_indices.items():
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, shuffle=True
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
