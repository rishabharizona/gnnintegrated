from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    Create a curriculum-based DataLoader using validation loss per domain to select 'easier' domains first.
    """
    # Group validation indices by domain
    domain_indices = {}
    for idx in range(len(val_dataset)):
        domain = val_dataset[idx][2]
        domain_indices.setdefault(domain, []).append(idx)

    domain_losses = []

    for domain, indices in domain_indices.items():
        subset = Subset(val_dataset, indices)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)

        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = tuple(item.cuda() for item in batch)
            output = algorithm.forward(batch)
            total_loss += output['class'].item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        domain_losses.append((domain, avg_loss))

    # Sort by easiest domains first
    domain_losses.sort(key=lambda x: x[1])
    
    # Print after sorting (to show domain difficulty order)
    print("\n--- Domain Ranking by Difficulty (easiest to hardest) ---")
    for rank, (domain, loss) in enumerate(domain_losses, 1):
        print(f"{rank}. Domain {domain}: Avg Val Loss = {loss:.4f}")

    # Curriculum progress
    num_domains = len(domain_losses)
    progress = min(1.0, ((stage + 1) / args.CL_PHASE_EPOCHS) ** 0.5)
    num_selected = max(1, int(np.ceil(progress * num_domains)))
    selected_domains = [domain for domain, _ in domain_losses[:num_selected]]

    # Select training indices from the chosen domains
    train_domain_indices = {}
    for idx in range(len(train_dataset)):
        domain = train_dataset[idx][2]
        train_domain_indices.setdefault(domain, []).append(idx)

    selected_indices = []
    for domain in selected_domains:
        selected_indices.extend(train_domain_indices.get(domain, []))

    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.N_WORKERS)

    return curriculum_loader
    
    
def split_dataset_by_domain(dataset, val_ratio=0.2, seed=42):
    domain_indices = {}

    for idx in range(len(dataset)):
        domain = dataset[idx][2]  # assumes domain is at index 2
        domain_indices.setdefault(domain, []).append(idx)

    train_indices, val_indices = [], []

    for domain, indices in domain_indices.items():
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, shuffle=True
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
    
    
class SubsetWithLabelSetter(Subset):
    def set_labels_by_index(self, labels, indices, key):
        if hasattr(self.dataset, 'set_labels_by_index'):
            self.dataset.set_labels_by_index(labels, indices, key)
