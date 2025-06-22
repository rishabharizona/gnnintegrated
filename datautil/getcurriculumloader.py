from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
import random

class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        self.dataset.set_labels_by_index(labels, indices, key)

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    domain_indices = {}
    for idx in range(len(val_dataset)):
        domain = val_dataset[idx][2]
        domain_indices.setdefault(domain, []).append(idx)

    domain_metrics = []
    algorithm.eval()
    with torch.no_grad():
        for domain, indices in domain_indices.items():
            subset = Subset(val_dataset, indices)
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)

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
    min_loss, max_loss = min(losses), max(losses)
    accs = [m[2] for m in domain_metrics]
    min_acc, max_acc = min(accs), max(accs)

    domain_scores = []
    for domain, loss, acc in domain_metrics:
        norm_loss = (loss - min_loss) / (max_loss - min_loss + 1e-10)
        norm_acc = (acc - min_acc) / (max_acc - min_acc + 1e-10)
        difficulty = 0.7 * norm_loss + 0.3 * (1 - norm_acc)
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

    if len(domain_scores) > num_selected:
        random_hard_domain = random.choice(domain_scores[num_selected:])[0]
        selected_domains.append(random_hard_domain)
        print(f"Adding random harder domain: {random_hard_domain}")

    train_domain_indices = {}
    for idx in range(len(train_dataset)):
        domain = train_dataset[idx][2]
        train_domain_indices.setdefault(domain, []).append(idx)

    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_indices = train_domain_indices[domain]
            n_samples = min(len(domain_indices), max(50, int(len(domain_indices) * 0.8)))
            selected_indices.extend(random.sample(domain_indices, n_samples))

    print(f"Selected {len(selected_indices)} samples from {len(selected_domains)} domains")
    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS, drop_last=True)
    return curriculum_loader

def get_samplewise_curriculum_loader(train_dataset, algorithm, stage, total_stages=5, alpha=1.0):
    scores = []
    loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    algorithm.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.cuda().float()
            y = y.cuda().long()
            logits = algorithm.predict(x)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction='none')
            for i in range(len(x)):
                scores.append((loss[i].item(), idx * 256 + i))

    scores.sort(key=lambda x: x[0])
    progress = (stage + 1) / total_stages
    cutoff = int(len(scores) * progress)
    selected_indices = [i for _, i in scores[:cutoff]]

    print(f"[Sample-wise CL] Stage {stage}, selected top {cutoff} samples")
    subset = Subset(train_dataset, selected_indices)
    return DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)

def split_dataset_by_domain(dataset, val_ratio=0.2, seed=42):
    domain_indices = {}
    for idx in range(len(dataset)):
        domain = dataset[idx][2]
        domain_indices.setdefault(domain, []).append(idx)

    train_indices, val_indices = [], []
    for domain, indices in domain_indices.items():
        train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed, shuffle=True)
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
