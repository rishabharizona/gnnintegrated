import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people
from torch_geometric.data import Batch
from datautil.graph_utils import convert_to_graph

# Task mapping for activity recognition
task_act = {'cross_people': cross_people}

class SubsetWithLabelSetter(Subset):
    """Subset with label setting capability"""
    def __init__(self, dataset, indices, domain_label=None):
        super().__init__(dataset, indices)
        self.domain_label = domain_label
        # Create mapping from subset index to original index
        self.index_map = {subset_idx: orig_idx for subset_idx, orig_idx in enumerate(indices)}
        
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        if self.domain_label is not None:
            return (data[0], data[1], self.domain_label, idx)  # Return subset index
        return data + (idx,)  # Return subset index as last element
        
    def set_labels_by_index(self, labels, subset_indices, key):
        """Set labels using subset indices"""
        # Convert subset indices to original dataset indices
        absolute_indices = [self.indices[i] for i in subset_indices]
        self.dataset.set_labels_by_index(labels, absolute_indices, key)

# Add this class to fix the index issue
class SafeSubset(Subset):
    """Safe subset that tracks its own indices"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Create mapping from subset index to original index
        self.index_map = {subset_idx: orig_idx for subset_idx, orig_idx in enumerate(indices)}
        
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return (*data, idx)  # Return subset index as last element


def get_gnn_dataloader(dataset, batch_size, num_workers, shuffle=True):
    """Create GNN-specific data loader"""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=shuffle
    )

def get_dataloader(args, tr, val, tar):
    """
    Create data loaders for training, validation, and target datasets
    Args:
        args: Configuration arguments
        tr: Training dataset
        val: Validation dataset
        tar: Target dataset
    Returns:
        Tuple of DataLoader objects
    """
    # ======= GNN-SPECIFIC LOADERS =======
    if hasattr(args, 'model_type') and args.model_type == 'gnn':
        train_loader = get_gnn_dataloader(
            tr, args.batch_size, args.N_WORKERS, shuffle=True)
        
        train_loader_noshuffle = get_gnn_dataloader(
            tr, args.batch_size, args.N_WORKERS, shuffle=False)
        
        valid_loader = get_gnn_dataloader(
            val, args.batch_size, args.N_WORKERS, shuffle=False)
        
        target_loader = get_gnn_dataloader(
            tar, args.batch_size, args.N_WORKERS, shuffle=False)
        
        return train_loader, train_loader_noshuffle, valid_loader, target_loader
    
    # ======= ORIGINAL LOADERS =======
    train_loader = DataLoader(
        dataset=tr, 
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS, 
        drop_last=False, 
        shuffle=True
    )
    
    train_loader_noshuffle = DataLoader(
        dataset=tr, 
        batch_size=args.batch_size, 
        num_workers=args.N_WORKERS, 
        drop_last=False, 
        shuffle=False
    )
    
    valid_loader = DataLoader(
        dataset=val, 
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS, 
        drop_last=False, 
        shuffle=False
    )
    
    target_loader = DataLoader(
        dataset=tar, 
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS, 
        drop_last=False, 
        shuffle=False
    )
    
    return train_loader, train_loader_noshuffle, valid_loader, target_loader

def get_act_dataloader(args):
    """
    Prepare activity recognition datasets and data loaders
    Args:
        args: Configuration arguments
    Returns:
        Tuple of DataLoader objects and datasets
    """
    source_datasetlist = []
    target_datalist = []
    pcross_act = task_act[args.task]

    # Get people configuration for the dataset
    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)
    
    # Create datasets for each person group
    for i, item in enumerate(tmpp):
        # ===== GNN TRANSFORM SELECTION =====
        if hasattr(args, 'model_type') and args.model_type == 'gnn':
            transform = actutil.act_to_graph_transform(args)
        else:
            transform = actutil.act_train()
        
        tdata = pcross_act.ActList(
            args, 
            args.dataset, 
            args.data_dir, 
            item, 
            i, 
            transform=transform
        )
        
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)
            # Adjust steps per epoch if needed
            if len(tdata) / args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata) / args.batch_size
    
    # Split source data into train/validation
    rate = 0.2  # Validation split ratio
    args.steps_per_epoch = int(args.steps_per_epoch * (1 - rate))
    
    # Combine source datasets
    tdata = combindataset(args, source_datasetlist)
    l = len(tdata.labels)
    indexall = np.arange(l)
    
    # Shuffle indices for train/validation split
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l * rate)
    indextr, indexval = indexall[ted:], indexall[:ted]
    
    # Create train and validation subsets using SafeSubset
    tr = SafeSubset(tdata, indextr)
    val = SafeSubset(tdata, indexval)
    
    # Combine target datasets
    targetdata = combindataset(args, target_datalist)
    
    # Create data loaders
    loaders = get_dataloader(args, tr, val, targetdata)
    return (*loaders, tr, val, targetdata)

def get_shap_batch(loader, size=100):
    """
    Extract a batch of data for SHAP analysis
    Args:
        loader: DataLoader to extract from
        size: Number of samples to extract
    Returns:
        Concatenated tensor of input samples
    """
    X_val = []
    for batch in loader:
        # Extract inputs from batch (could be tuple or tensor)
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        X_val.append(inputs)
        
        # Stop when we have enough samples
        if len(torch.cat(X_val)) >= size:
            break
    
    # Return exactly size samples
    return torch.cat(X_val)[:size]

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    Create a curriculum data loader based on domain difficulty
    Args:
        args: Configuration arguments
        algorithm: Model for domain evaluation
        train_dataset: Full training dataset
        val_dataset: Validation dataset
        stage: Current training stage/phase
    Returns:
        Curriculum DataLoader with selected samples
    """
    # Group validation indices by domain
    domain_indices = {}
    for idx in range(len(val_dataset)):
        domain = val_dataset[idx][2]  # Assuming domain is at index 2
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

    # Calculate domain difficulty scores with numerical stability
    losses = [m[1] for m in domain_metrics]
    min_loss, max_loss = min(losses), max(losses)
    accs = [m[2] for m in domain_metrics]
    min_acc, max_acc = min(accs), max(accs)
    
    domain_scores = []
    for domain, loss, acc in domain_metrics:
        # Safe normalization with epsilon protection
        loss_range = max_loss - min_loss
        acc_range = max_acc - min_acc
        
        norm_loss = 0.0
        if loss_range > 1e-8:  # Only normalize if significant range exists
            norm_loss = (loss - min_loss) / loss_range
        
        norm_acc = 0.0
        if acc_range > 1e-8:
            norm_acc = (acc - min_acc) / acc_range
        
        # Clamp values to [0,1] range
        norm_loss = max(0.0, min(1.0, norm_loss))
        norm_acc = max(0.0, min(1.0, norm_acc))
        
        # Difficulty score: higher = harder
        difficulty = 0.7 * norm_loss + 0.3 * (1 - norm_acc)
        domain_scores.append((domain, difficulty))

    # Sort by easiest domains first
    domain_scores.sort(key=lambda x: x[1])
    
    print("\n--- Domain Difficulty Ranking (easiest to hardest) ---")
    for rank, (domain, difficulty) in enumerate(domain_scores, 1):
        print(f"{rank}. Domain {domain}: Difficulty = {difficulty:.4f}")

    # Curriculum progression
    num_domains = len(domain_scores)
    progress = min(1.0, (stage + 1) / args.CL_PHASE_EPOCHS)
    progress = np.sqrt(progress)  # Slower initial progression
    
    num_selected = max(2, min(num_domains, int(np.ceil(progress * num_domains * 0.8))))
    selected_domains = [domain for domain, _ in domain_scores[:num_selected]]
    
    # Add random harder domain for diversity
    if len(domain_scores) > num_selected:
        random_hard_domain = random.choice(domain_scores[num_selected:])[0]
        selected_domains.append(random_hard_domain)
        print(f"Adding random harder domain: {random_hard_domain}")

    # Gather training indices from selected domains
    train_domain_indices = {}
    max_domain_size = 0
    for idx in range(len(train_dataset)):
        domain = train_dataset[idx][2]  # Assuming domain is at index 2
        train_domain_indices.setdefault(domain, []).append(idx)
        if len(train_domain_indices[domain]) > max_domain_size:
            max_domain_size = len(train_domain_indices[domain])

    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_indices = train_domain_indices[domain]
            # Proportional sampling based on domain size
            sample_ratio = 0.5 + 0.5 * (1 - len(domain_indices) / max_domain_size)
            n_samples = min(len(domain_indices), max(50, int(len(domain_indices) * sample_ratio)))
            selected_indices.extend(random.sample(domain_indices, n_samples))

    print(f"Selected {len(selected_indices)} samples from {len(selected_domains)} domains")
    
    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices, domain_label=-1)
    
    # ===== GNN-SPECIFIC CURRICULUM LOADER =====
    if hasattr(args, 'model_type') and args.model_type == 'gnn':
        curriculum_loader = DataLoader(
            curriculum_subset, 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=args.N_WORKERS,
            drop_last=True
        )
    else:
        curriculum_loader = DataLoader(
            curriculum_subset, 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=args.N_WORKERS,
            drop_last=True
        )

    return curriculum_loader

def split_dataset_by_domain(dataset, val_ratio=0.2, seed=42):
    """
    Split dataset into train/validation by domain
    Args:
        dataset: Dataset to split
        val_ratio: Validation set ratio
        seed: Random seed
    Returns:
        Tuple of train and validation subsets
    """
    domain_indices = {}
    for idx in range(len(dataset)):
        domain = dataset[idx][2]  # Assuming domain is at index 2
        domain_indices.setdefault(domain, []).append(idx)

    train_indices, val_indices = [], []
    for domain, indices in domain_indices.items():
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, shuffle=True
        )
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
    
def get_shap_batch(loader, size=100):
    """
    Extract a batch of data for SHAP analysis
    Args:
        loader: DataLoader to extract from
        size: Number of samples to extract
    Returns:
        Concatenated tensor of input samples
    """
    X_val = []
    for batch in loader:
        # Extract inputs from batch (could be tuple or tensor)
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        X_val.append(inputs)
        
        # Stop when we have enough samples
        if len(torch.cat(X_val)) >= size:
            break
    
    # Return exactly size samples
    return torch.cat(X_val)[:size]
