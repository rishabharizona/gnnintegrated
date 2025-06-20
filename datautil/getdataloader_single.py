import numpy as np
from torch.utils.data import DataLoader
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people

# Task mapping for activity recognition
task_act = {'cross_people': cross_people}

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
        tdata = pcross_act.ActList(
            args, 
            args.dataset, 
            args.data_dir, 
            item, 
            i, 
            transform=actutil.act_train()
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
    
    # Create train and validation subsets
    tr = subdataset(args, tdata, indextr)
    val = subdataset(args, tdata, indexval)
    
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
