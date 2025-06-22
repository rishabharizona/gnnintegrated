import torch
import numpy as np
import random
from torch.utils.data import Subset, DataLoader

class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        self.dataset.set_labels_by_index(labels, indices, key)

def get_samplewise_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """Constructs a curriculum dataloader by selecting easy samples based on individual sample losses."""
    algorithm.eval()
    sample_losses = []

    # Create a loader for the validation dataset to estimate sample difficulty
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)

    print("[INFO] Evaluating sample-level difficulty from validation data...")
    with torch.no_grad():
        base_idx = 0
        for batch in val_loader:
            x, y, _ = batch
            x, y = x.cuda().float(), y.cuda().long()
            outputs = algorithm.predict(x)
            losses = torch.nn.functional.cross_entropy(outputs, y, reduction='none')
            for i, l in enumerate(losses):
                sample_losses.append((base_idx + i, l.item()))
            base_idx += x.size(0)

    # Sort samples by loss (ascending = easiest first)
    sample_losses.sort(key=lambda x: x[1])

    # Curriculum progression: slowly introduce harder examples
    progress = min(1.0, (stage + 1) / args.CL_PHASE_EPOCHS)
    num_selected = int(len(sample_losses) * progress)
    easy_indices = [idx for idx, _ in sample_losses[:num_selected]]

    # Introduce a few hard samples (top 10%) for diversity
    hard_pool = sample_losses[-int(0.1 * len(sample_losses)):] if len(sample_losses) > 10 else []
    hard_samples = random.sample(hard_pool, min(10, len(hard_pool)))
    hard_indices = [idx for idx, _ in hard_samples]

    selected_indices = list(set(easy_indices + hard_indices))

    print(f"\n📚 Curriculum Stage {stage+1}: Selected {len(selected_indices)} samples (Progress: {progress*100:.1f}%)")

    # Create DataLoader from selected samples
    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    curriculum_loader = DataLoader(
        curriculum_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.N_WORKERS,
        drop_last=True
    )

    return curriculum_loader

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
