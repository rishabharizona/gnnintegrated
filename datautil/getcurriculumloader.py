from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
import random
from collections import defaultdict

class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        self.dataset.set_labels_by_index(labels, indices, key)

def split_dataset_by_domain(dataset, val_ratio=0.2, seed=42):
    domain_indices = {}
    for idx in range(len(dataset)):
        domain = dataset[idx][2]
        domain_indices.setdefault(domain, []).append(idx)

    train_indices, val_indices = [], []
    for domain, indices in domain_indices.items():
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, shuffle=True
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    if stage < getattr(args, 'CURRICULUM_WARMUP_EPOCHS', 5):
        print("Warm-up phase: using full training dataset.")
        return DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.N_WORKERS, drop_last=True)

    domain_indices = {}
    for idx in range(len(val_dataset)):
        domain = val_dataset[idx][2]
        domain_indices.setdefault(domain, []).append(idx)

    domain_metrics = []
    algorithm.eval()
    with torch.no_grad():
        for domain, indices in domain_indices.items():
            subset = Subset(val_dataset, indices)
            loader = DataLoader(subset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.N_WORKERS)

            total_loss, correct, total, num_batches = 0.0, 0, 0, 0
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

    losses = [m[1] for m in domain_metrics]
    accs = [m[2] for m in domain_metrics]
    mean_loss, std_loss = np.mean(losses), np.std(losses) + 1e-10
    mean_acc, std_acc = np.mean(accs), np.std(accs) + 1e-10

    domain_scores = []
    for domain, loss, acc in domain_metrics:
        norm_loss = (loss - mean_loss) / std_loss
        norm_acc = (acc - mean_acc) / std_acc
        difficulty = 0.8 * norm_loss - 0.2 * norm_acc  # give more weight to loss
        domain_scores.append((domain, difficulty))

    domain_scores.sort(key=lambda x: x[1])

    print("\n--- Domain Difficulty Ranking (easiest to hardest) ---")
    for rank, (domain, difficulty) in enumerate(domain_scores, 1):
        print(f"{rank}. Domain {domain}: Difficulty = {difficulty:.4f}")

    # Use smooth curriculum: increase number of hard domains gradually
    num_domains = len(domain_scores)
    total_curriculum_epochs = getattr(args, 'CL_PHASE_EPOCHS', 10)
    progress = min(1.0, (stage + 1) / total_curriculum_epochs)
    progress = np.power(progress, 2.5)  # More slow progression

    easy_pct = max(0.2, 1.0 - progress)  # Start from 100% easy, reduce over time
    num_easy = max(1, int(easy_pct * num_domains))
    selected_domains = [domain for domain, _ in domain_scores[:num_easy]]

    # Gradually introduce mid and hard domains
    if progress > 0.3:
        mid_start = num_easy
        mid_end = min(num_domains, mid_start + int(num_domains * 0.3))
        selected_domains += [domain for domain, _ in domain_scores[mid_start:mid_end]]
    if progress > 0.6:
        selected_domains += [domain for domain, _ in domain_scores[mid_end:]]

    selected_domains = list(set(selected_domains))

    train_domain_indices = defaultdict(list)
    for idx in range(len(train_dataset)):
        domain = train_dataset[idx][2]
        train_domain_indices[domain].append(idx)

    domain_difficulty_dict = dict(domain_scores)
    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_idxs = train_domain_indices[domain]
            label_indices = defaultdict(list)
            for idx in domain_idxs:
                label = train_dataset[idx][1]
                label_indices[label].append(idx)

            difficulty = domain_difficulty_dict.get(domain, 0.5)
            retain_ratio = max(0.3, 1.0 - 0.5 * difficulty)
            n_samples = int(retain_ratio * len(domain_idxs))

            samples_per_class = max(1, n_samples // len(label_indices))
            for label, idxs in label_indices.items():
                sampled = random.sample(idxs, min(len(idxs), samples_per_class))
                selected_indices.extend(sampled)

    print(f"Selected {len(selected_indices)} samples from {len(selected_domains)} domains")

    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS,
                                   drop_last=True)

    return curriculum_loader
