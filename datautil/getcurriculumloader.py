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
        difficulty = 0.7 * norm_loss - 0.3 * norm_acc
        domain_scores.append((domain, difficulty))

    domain_scores.sort(key=lambda x: x[1])

    print("\n--- Domain Difficulty Ranking (easiest to hardest) ---")
    for rank, (domain, difficulty) in enumerate(domain_scores, 1):
        print(f"{rank}. Domain {domain}: Difficulty = {difficulty:.4f}")

    num_domains = len(domain_scores)
    progress = min(1.0, (stage + 1) / args.CL_PHASE_EPOCHS)
    progress = np.sqrt(progress)
    num_selected = max(2, min(num_domains, int(np.ceil(progress * num_domains * 0.8))))
    selected_domains = [domain for domain, _ in domain_scores[:num_selected]]

    if stage > args.CL_PHASE_EPOCHS // 2 and len(domain_scores) > num_selected:
        hardest_domain = domain_scores[-1][0]
        selected_domains.append(hardest_domain)
        print(f"Adding hardest domain for challenge: {hardest_domain}")

    if not hasattr(args, 'selected_domain_history'):
        args.selected_domain_history = set()
    args.selected_domain_history.update(selected_domains)
    selected_domains = list(args.selected_domain_history)

    train_domain_indices = {}
    for idx in range(len(train_dataset)):
        domain = train_dataset[idx][2]
        train_domain_indices.setdefault(domain, []).append(idx)

    domain_difficulty_dict = dict(domain_scores)
    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_idxs = train_domain_indices[domain]
            label_indices = defaultdict(list)
            for idx in domain_idxs:
                label = train_dataset[idx][1]
                label_indices[label].append(idx)

            weight = 1 - domain_difficulty_dict.get(domain, 0.5)
            n_samples = int(min(len(domain_idxs), max(50, len(domain_idxs) * (0.5 + weight))))
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
