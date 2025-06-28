import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch
import datautil.graph_utils as graph_utils
from typing import List, Tuple, Dict, Any, Optional
import collections
import torch.nn.functional as F
from collections import defaultdict

# Task mapping for activity recognition
task_act = {'cross_people': cross_people}

class ConsistentFormatWrapper(Dataset):
    """Ensures samples always return (graph, label, domain) format"""
    def __init__(self, dataset):
        if isinstance(dataset, DataLoader):
            raise TypeError("ConsistentFormatWrapper requires a Dataset, not a DataLoader")
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
        if isinstance(dataset, DataLoader):
            raise TypeError("SafeSubset requires a Dataset, not a DataLoader")
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
            # FIXED: Use keys() method instead of keys attribute
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
    if isinstance(dataset, DataLoader):
        raise TypeError("get_gnn_dataloader requires a Dataset, not a DataLoader")
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
    
    # Ensure we have datasets, not loaders
    for i, ds in enumerate([tr, val, tar]):
        if isinstance(ds, DataLoader):
            raise TypeError(f"get_dataloader requires Datasets, not DataLoaders (argument {i})")
    
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
    """Advanced curriculum data loader with multi-faceted difficulty scoring"""
    # Get difficulty threshold for current stage
    if stage < len(args.CL_DIFFICULTY):
        difficulty_threshold = args.CL_DIFFICULTY[stage]
    else:
        difficulty_threshold = 1.0  # Use all samples if stage exceeds threshold list
    
    # Helper function to detect graph data format
    def is_graph_data(dataset):
        if len(dataset) == 0:
            return False
        sample = dataset[0]
        return (
            (isinstance(sample, tuple) and len(sample) >= 3 and isinstance(sample[0], Data)) or
            isinstance(sample, Data) or
            (isinstance(sample, dict) and 'graph' in sample)
        )
    
    # Ensure we have datasets, not loaders
    if isinstance(train_dataset, DataLoader):
        raise TypeError("get_curriculum_loader requires train Dataset, not DataLoader")
    if isinstance(val_dataset, DataLoader):
        raise TypeError("get_curriculum_loader requires val Dataset, not DataLoader")

    # Create helper functions for data extraction
    def get_domain(sample):
        """Extract domain from different data formats"""
        if isinstance(sample, tuple) and len(sample) >= 3:
            return sample[2]
        elif isinstance(sample, Data) and hasattr(sample, 'domain'):
            return sample.domain
        elif isinstance(sample, dict) and 'domain' in sample:
            return sample['domain']
        return 0  # Default domain if not found
    
    def to_tensor(domain):
        """Convert domain to tensor if needed"""
        return domain.item() if isinstance(domain, torch.Tensor) else domain
    
    # ================= ADVANCED DIFFICULTY SCORING =================
    # Step 1: Calculate domain divergence metrics
    domain_features = {}
    with torch.no_grad():
        algorithm.eval()
        
        # Extract features for all validation samples
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            
            # Extract data based on format
            if is_graph_data([sample]):
                inputs = sample[0] if isinstance(sample, tuple) else sample
                labels = sample[1] if isinstance(sample, tuple) else None
            else:
                inputs = sample[0] if isinstance(sample, tuple) else sample
                labels = sample[1] if isinstance(sample, tuple) else None
            
            # Move to device and get features
            inputs = inputs.to(args.device)
            features = algorithm.featurizer(inputs).detach().cpu().numpy()
            domain = to_tensor(get_domain(sample))
            
            if domain not in domain_features:
                domain_features[domain] = []
            domain_features[domain].append(features)
        
        # Compute domain centroids
        domain_centroids = {}
        for domain, features in domain_features.items():
            domain_centroids[domain] = np.mean(features, axis=0)
        
        # Compute pairwise domain distances
        domain_distances = {}
        domains = list(domain_centroids.keys())
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                dist = np.linalg.norm(domain_centroids[domain1] - domain_centroids[domain2])
                domain_distances[(domain1, domain2)] = dist
                domain_distances[(domain2, domain1)] = dist
                
        # Compute average distance for each domain
        domain_avg_dist = {}
        for domain in domains:
            distances = [dist for (d1, d2), dist in domain_distances.items() 
                         if d1 == domain or d2 == domain]
            domain_avg_dist[domain] = np.mean(distances) if distances else 0

    # Step 2: Calculate per-sample difficulty using multiple metrics
    sample_difficulties = []
    with torch.no_grad():
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            
            # Improved label extraction
            if is_graph_data([sample]):
                if isinstance(sample, tuple):
                    inputs, labels = sample[0], sample[1]
                else:
                    inputs = sample
                    # Handle PyG Data objects
                    labels = sample.y if hasattr(sample, 'y') else None
            else:
                if isinstance(sample, tuple) and len(sample) >= 2:
                    inputs, labels = sample[0], sample[1]
                else:
                    inputs = sample
                    labels = None
            
            # Ensure labels are properly formatted
            if labels is not None:
                if isinstance(labels, torch.Tensor) and labels.dim() == 0:
                    labels = labels.unsqueeze(0)  # Convert scalar to tensor
                
            # Move to device
            inputs = inputs.to(args.device)
            if labels is not None:
                labels = labels.to(args.device).long() if isinstance(labels, torch.Tensor) else labels
            
            # Get model predictions
            outputs = algorithm.predict(inputs)
            probs = F.softmax(outputs, dim=-1)
            
            # Add dimension check before loss calculation
            valid_labels = labels is not None and isinstance(labels, torch.Tensor) and labels.numel() > 0
            
            if valid_labels and outputs.shape[0] == labels.shape[0]:
                # Cross-entropy loss
                loss = F.cross_entropy(outputs, labels).item()
                
                # Accuracy (0-1)
                _, preds = torch.max(outputs, 1)
                accuracy = (preds == labels).float().mean().item()
            else:
                loss = 0
                accuracy = 0
                if args.verbose:
                    print(f"Warning: Invalid labels in sample {idx} - "
                          f"Output shape: {outputs.shape}, Label shape: {labels.shape if labels is not None else 'None'}")
                
            # Confidence-based metrics
            max_prob = probs.max().item()
            entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
            
            # Domain-based difficulty
            domain = to_tensor(get_domain(sample))
            domain_diff = domain_avg_dist.get(domain, 0)
            
            # Combined difficulty score (higher = more difficult)
            difficulty_score = (
                0.4 * loss + 
                0.3 * (1 - accuracy) + 
                0.1 * domain_diff + 
                0.1 * entropy + 
                0.1 * (1 - max_prob)
            )
            
            sample_difficulties.append((idx, difficulty_score))
    
    # Normalize difficulties to [0, 1] range
    difficulties = [d for _, d in sample_difficulties]
    min_difficulty = min(difficulties)
    max_difficulty = max(difficulties) if max(difficulties) > min_difficulty else 1
    normalized_difficulties = [
        (d - min_difficulty) / (max_difficulty - min_difficulty) 
        for d in difficulties
    ]
    
    # Update sample difficulties with normalized values
    sample_difficulties = [(idx, norm_d) for (idx, _), norm_d in 
                          zip(sample_difficulties, normalized_difficulties)]
    
    # ================= DYNAMIC THRESHOLD ADJUSTMENT =================
    # Adjust threshold based on model performance (easier if model struggling)
    if stage > 0 and logs['valid_acc'] and logs['valid_acc'][-1] < 50:  # Low accuracy
        difficulty_threshold = max(0.1, difficulty_threshold * 0.8)
        print(f"Model struggling (acc={logs['valid_acc'][-1]:.1f}%), "
              f"lowering threshold to {difficulty_threshold:.2f}")
    
    # ================= SAMPLE SELECTION STRATEGY =================
    # Select samples below difficulty threshold
    selected_indices = [idx for idx, diff in sample_difficulties if diff <= difficulty_threshold]
    
    # If too few samples, expand threshold
    min_samples = int(0.2 * len(train_dataset))  # At least 20% of dataset
    if len(selected_indices) < min_samples:
        # Find the easiest min_samples
        sample_difficulties.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in sample_difficulties[:min_samples]]
        print(f"Expanding curriculum to {len(selected_indices)} easiest samples")
    
    # Confidence-based filtering (remove highly uncertain samples)
    confidence_threshold = 0.7 - (stage * 0.1)  # Decreases with stages
    high_confidence_indices = []
    for idx in selected_indices:
        sample = train_dataset[idx]
        inputs = sample[0] if isinstance(sample, tuple) else sample
        inputs = inputs.to(args.device)
        outputs = algorithm.predict(inputs)
        probs = F.softmax(outputs, dim=-1)
        max_prob = probs.max().item()
        
        if max_prob >= confidence_threshold:
            high_confidence_indices.append(idx)
    
    if high_confidence_indices:
        selected_indices = high_confidence_indices
        print(f"Filtered to {len(selected_indices)} high-confidence samples "
              f"(confidence ≥ {confidence_threshold:.2f})")
    
    # ================= DOMAIN BALANCING =================
    # Ensure we have representation from all domains
    domain_counts = defaultdict(int)
    domain_indices = defaultdict(list)
    
    for idx in selected_indices:
        sample = train_dataset[idx]
        domain = to_tensor(get_domain(sample))
        domain_counts[domain] += 1
        domain_indices[domain].append(idx)
    
    # Redistribute to balance domains
    if domain_counts:
        min_domain_count = min(domain_counts.values())
        balanced_indices = []
        
        for domain, indices in domain_indices.items():
            # If domain has more than min, randomly select min samples
            if len(indices) > min_domain_count:
                balanced_indices.extend(np.random.choice(
                    indices, min_domain_count, replace=False))
            else:
                balanced_indices.extend(indices)
        
        # Only use balanced set if it has sufficient size
        if len(balanced_indices) >= min_samples:
            selected_indices = balanced_indices
            print(f"Balanced curriculum: {len(domain_counts)} domains, "
                  f"{len(selected_indices)} samples ({min_domain_count} per domain)")
    
    # ================= FINAL SELECTION =================
    # Create curriculum subset
    curriculum_subset = SafeSubset(train_dataset, selected_indices)
    
    # Calculate average difficulty of selected samples
    avg_difficulty = np.mean([diff for idx, diff in sample_difficulties 
                             if idx in selected_indices])
    
    print(f"Curriculum Stage {stage+1}: "
          f"Threshold={difficulty_threshold:.2f}, "
          f"Samples={len(selected_indices)}/{len(train_dataset)} "
          f"({len(selected_indices)/len(train_dataset):.1%}), "
          f"Avg Difficulty={avg_difficulty:.3f}")
    
    return curriculum_subset

def get_shap_batch(loader, size=100):
    """Extract a batch of data for SHAP analysis"""
    samples = []
    for batch in loader:
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        if isinstance(inputs, (Data, Batch)):
            # Handle PyG DataBatch objects
            samples.append(inputs)
        elif isinstance(inputs, torch.Tensor):
            # Handle standard tensors
            samples.append(inputs)
        else:
            # Handle other formats
            try:
                samples.append(torch.tensor(inputs))
            except:
                print(f"Warning: Could not convert input of type {type(inputs)} to tensor")
                
        if len(samples) * inputs.size(0) >= size:
            break
            
    # Return first batch if we have PyG Data objects
    if isinstance(samples[0], (Data, Batch)):
        return samples[0]
    
    # Concatenate tensors
    return torch.cat(samples)[:size]
