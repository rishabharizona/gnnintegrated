import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch
import datautil.graph_utils as graph_utils
from typing import List, Tuple, Dict, Any, Optional
import collections

# Task mapping for activity recognition
task_act = {'cross_people': cross_people}

class ConsistentFormatWrapper(torch.utils.data.Dataset):
    """Ensures samples always return (graph, label, domain) format"""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Convert to consistent (graph, label, domain) format
        if isinstance(sample, tuple) and len(sample) >= 3:
            return sample[0], sample[1], sample[2]
        elif isinstance(sample, Data):
            return (
                sample, 
                sample.y if hasattr(sample, 'y') else 0,
                sample.domain if hasattr(sample, 'domain') else 0
            )
        elif isinstance(sample, dict) and 'graph' in sample:
            return (
                sample['graph'],
                sample.get('label', 0),
                sample.get('domain', 0)
            )
        else:
            return sample, 0, 0
    
    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class SafeSubset(Subset):
    """Safe subset that converts numpy types to PyTorch tensors"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.indices = indices
        
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return self.convert_data(data)
    
    def convert_data(self, data):
        """Recursively convert numpy types to PyTorch-compatible formats"""
        if isinstance(data, tuple):
            return tuple(self.convert_data(x) for x in data)
        elif isinstance(data, list):
            return [self.convert_data(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        elif isinstance(data, np.generic):
            return data.item()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, Data):
            # Correctly handle PyG Data objects
            for key in data.keys():
                if hasattr(data, key):
                    setattr(data, key, self.convert_data(getattr(data, key)))
            return data
        else:
            return data

def collate_gnn(batch):
    """Robust collate function for GNN data"""
    graphs, labels, domains = [], [], []
    
    for sample in batch:
        if isinstance(sample, tuple) and len(sample) >= 3:
            graphs.append(sample[0])
            labels.append(sample[1])
            domains.append(sample[2])
        elif isinstance(sample, Data):
            graphs.append(sample)
            labels.append(sample.y if hasattr(sample, 'y') else 0)
            domains.append(sample.domain if hasattr(sample, 'domain') else 0)
        elif isinstance(sample, dict) and 'graph' in sample:
            graphs.append(sample['graph'])
            labels.append(sample.get('label', 0))
            domains.append(sample.get('domain', 0))
        else:
            graphs.append(sample)
            labels.append(0)
            domains.append(0)
    
    batched_graph = Batch.from_data_list(graphs)
    labels = torch.tensor(labels, dtype=torch.long)
    domains = torch.tensor(domains, dtype=torch.long)
    
    return batched_graph, labels, domains

def get_gnn_dataloader(dataset, batch_size, num_workers, shuffle=True):
    """Create GNN-specific data loader with custom collate"""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=collate_gnn
    )

def get_dataloader(args, tr, val, tar):
    """Detect graph data and create appropriate loaders"""
    # Handle empty datasets
    if len(tr) == 0:
        raise ValueError("Training dataset is empty")
    
    sample = tr[0]
    is_graph_data = False
    
    # Check if the sample is a tuple containing a Data object
    if isinstance(sample, tuple) and len(sample) >= 3 and isinstance(sample[0], Data):
        is_graph_data = True
    # Check if the sample is a Data object itself
    elif isinstance(sample, Data):
        is_graph_data = True
    # Check if the sample is a dictionary with a graph key
    elif isinstance(sample, dict) and 'graph' in sample:
        is_graph_data = True

    if is_graph_data or (hasattr(args, 'model_type') and args.model_type == 'gnn'):
        return (
            get_gnn_dataloader(tr, args.batch_size, args.N_WORKERS, shuffle=True),
            get_gnn_dataloader(tr, args.batch_size, args.N_WORKERS, shuffle=False),
            get_gnn_dataloader(val, args.batch_size, args.N_WORKERS, shuffle=False),
            get_gnn_dataloader(tar, args.batch_size, args.N_WORKERS, shuffle=False)
        )
    
    # Standard data loaders
    return (
        DataLoader(tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=True),
        DataLoader(tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=False),
        DataLoader(val, batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=False),
        DataLoader(tar, batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=False)
    )

def get_act_dataloader(args):
    """Create activity recognition data loaders"""
    source_datasets = []
    target_datasets = []
    pcross_act = task_act[args.task]
    
    if args.dataset not in args.act_people:
        raise ValueError(f"Dataset {args.dataset} not found in act_people configuration")
    
    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)
    
    for i, item in enumerate(tmpp):
        # Use graph transforms for GNN models
        transform = actutil.act_to_graph_transform(args) if (
            hasattr(args, 'model_type') and args.model_type == 'gnn') else actutil.act_train()
        
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=transform)
        
        if i in args.test_envs:
            target_datasets.append(tdata)
        else:
            source_datasets.append(tdata)
    
    # Combine and wrap datasets
    source_data = combindataset(args, source_datasets)
    source_data = ConsistentFormatWrapper(source_data)
    target_data = combindataset(args, target_datasets)
    target_data = ConsistentFormatWrapper(target_data)
    
    # Split source data (80% train, 20% validation)
    l = len(source_data)
    if l == 0:
        raise ValueError("Source dataset is empty after processing")
    
    indices = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    split_point = int(l * 0.8)
    train_indices = indices[:split_point]  # First 80% for training
    val_indices = indices[split_point:]    # Last 20% for validation
    
    train_set = SafeSubset(source_data, train_indices)
    val_set = SafeSubset(source_data, val_indices)
    
    # Create data loaders
    loaders = get_dataloader(args, train_set, val_set, target_data)
    return (*loaders, train_set, val_set, target_data)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """Create curriculum data loader based on domain difficulty"""
    # Collect domain indices from validation set
    domain_indices = {}
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        domain = 0
        
        # Extract domain from different formats
        if isinstance(sample, tuple) and len(sample) >= 3:
            domain = sample[2]
        elif isinstance(sample, Data) and hasattr(sample, 'domain'):
            domain = sample.domain
        elif isinstance(sample, dict) and 'domain' in sample:
            domain = sample['domain']
            
        if isinstance(domain, torch.Tensor):
            domain = domain.item()
            
        domain_indices.setdefault(domain, []).append(idx)
    
    # If no domains found, return full dataset
    if not domain_indices:
        print("Warning: No domains found for curriculum learning, using full dataset")
        if hasattr(args, 'model_type') and args.model_type == 'gnn':
            return get_gnn_dataloader(train_dataset, args.batch_size, 0, True)
        else:
            return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Compute loss per domain
    domain_metrics = []
    with torch.no_grad():
        for domain, indices in domain_indices.items():
            subset = Subset(val_dataset, indices)
            is_graph_data = hasattr(args, 'model_type') and args.model_type == 'gnn'
            
            if is_graph_data:
                loader = get_gnn_dataloader(subset, args.batch_size, 0, False)
            else:
                loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
            
            total_loss = 0.0
            for batch in loader:
                if is_graph_data:
                    inputs, labels, _ = batch
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                
                output = algorithm.predict(inputs)
                loss = torch.nn.functional.cross_entropy(output, labels)
                total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(loader))
            domain_metrics.append((domain, avg_loss))
    
    # Calculate domain difficulty using only loss
    losses = [m[1] for m in domain_metrics]
    min_loss, max_loss = min(losses), max(losses)
    loss_range = max(1e-8, max_loss - min_loss)
    
    domain_scores = []
    for domain, loss in domain_metrics:
        difficulty = (loss - min_loss) / loss_range
        domain_scores.append((domain, difficulty))
    
    # Sort domains by difficulty (easiest first)
    domain_scores.sort(key=lambda x: x[1])
    
    # Curriculum progression
    total_stages = len(args.CL_PHASE_EPOCHS)
    progress = min(1.0, (stage + 1) / total_stages)
    num_selected = max(1, int(np.ceil(progress * len(domain_scores))))
    selected_domains = [domain for domain, _ in domain_scores[:num_selected]]
    
    # Collect training samples from selected domains
    selected_indices = []
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        domain = 0
        
        if isinstance(sample, tuple) and len(sample) >= 3:
            domain = sample[2]
        elif isinstance(sample, Data) and hasattr(sample, 'domain'):
            domain = sample.domain
        elif isinstance(sample, dict) and 'domain' in sample:
            domain = sample['domain']
            
        if isinstance(domain, torch.Tensor):
            domain = domain.item()
            
        if domain in selected_domains:
            selected_indices.append(idx)
    
    # Fallback to full dataset if no samples selected
    if not selected_indices:
        selected_indices = list(range(len(train_dataset)))
    
    # Create curriculum subset
    curriculum_subset = SafeSubset(train_dataset, selected_indices)
    
    # CRITICAL FIX: Use GNN collate function for graph data
    if hasattr(args, 'model_type') and args.model_type == 'gnn':
        return get_gnn_dataloader(curriculum_subset, args.batch_size, 0, True)
    else:
        # For non-graph data, use standard DataLoader
        return DataLoader(
            curriculum_subset, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=None  # Use default collate
        )

def get_shap_batch(loader, size=100):
    """Extract a batch of data for SHAP analysis"""
    samples = []
    for batch in loader:
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        samples.append(inputs)
        if len(torch.cat(samples)) >= size:
            break
    return torch.cat(samples)[:size]
