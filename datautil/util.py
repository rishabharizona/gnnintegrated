import numpy as np
import torch
from torch_geometric.data import Data  # Add this import

def Nmax(args, d):
    """
    Find the position index for a domain relative to test environments
    Args:
        args: Configuration arguments with test_envs attribute
        d: Domain index to evaluate
    Returns:
        Position index relative to test environments
    """
    for i in range(len(args.test_envs)):
        if d < args.test_envs[i]:
            return i
    return len(args.test_envs)

class basedataset(object):
    """Basic dataset class for simple x,y pairs"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class mydataset(object):
    """Advanced dataset class with support for multiple label types and transformations"""
    def __init__(self, args):
        self.x = None                 # Input features
        self.labels = None             # Class labels
        self.dlabels = None            # Domain labels
        self.pclabels = None           # Pseudo-class labels
        self.pdlabels = None           # Pseudo-domain labels
        self.task = None               # Task identifier
        self.dataset = None            # Dataset name
        self.transform = None          # Input transformations
        self.target_transform = None   # Label transformations
        self.loader = None             # Data loader reference
        self.args = args               # Configuration arguments

    def set_labels(self, tlabels=None, label_type='domain_label'):
        """
        Set labels for entire dataset
        Args:
            tlabels: Label values to set
            label_type: Type of labels to set (class_label, domain_label, etc.)
        """
        assert len(tlabels) == len(self.x)
        if label_type == 'pclabel':
            self.pclabels = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels = tlabels
        elif label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        """
        Set labels for specific indices
        Args:
            tlabels: Label values to set
            tindex: Indices to update
            label_type: Type of labels to set
        """
        if label_type == 'pclabel':
            self.pclabels[tindex] = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels
        elif label_type == 'domain_label':
            self.dlabels[tindex] = tlabels
        elif label_type == 'class_label':
            self.labels[tindex] = tlabels

    def target_trans(self, y):
        """Apply target transformation if defined"""
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        """Apply input transformation if defined"""
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        """Get item with all label types and transformations"""
        x = self.input_trans(self.x[index])
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        pctarget = self.target_trans(self.pclabels[index])
        pdtarget = self.target_trans(self.pdlabels[index])
        
        # For GNN models, convert to graph data object
        if hasattr(self.args, 'model_type') and self.args.model_type == 'gnn' and not isinstance(x, Data):
            x = Data(x=x, edge_index=torch.tensor([[], []]), edge_attr=torch.tensor([]))
        return x, ctarget, dtarget, pctarget, pdtarget, index

    def __len__(self):
        return len(self.x)

class subdataset(mydataset):
    """Subset of a dataset defined by indices"""
    def __init__(self, args, dataset, indices):
        super(subdataset, self).__init__(args)
        
        # Convert any index type to Python integers
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, (torch.Tensor, np.ndarray)):
            indices = indices.tolist()
        elif not isinstance(indices, list):
            indices = list(indices)
        
        # Convert to integer indices
        indices = [int(i) for i in indices]
        
        # Extract data using basic Python indexing
        self.x = dataset.x[indices]
        self.labels = dataset.labels[indices] if dataset.labels is not None else None
        self.dlabels = dataset.dlabels[indices] if dataset.dlabels is not None else None
        self.pclabels = dataset.pclabels[indices] if dataset.pclabels is not None else None
        self.pdlabels = dataset.pdlabels[indices] if dataset.pdlabels is not None else None
        
        self.loader = dataset.loader
        self.task = dataset.task
        self.dataset = dataset.dataset
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

class combindataset(mydataset):
    """Combined dataset from multiple source datasets"""
    def __init__(self, args, datalist):
        super(combindataset, self).__init__(args)
        self.domain_num = len(datalist)
        self.loader = datalist[0].loader
        
        # Combine all datasets in the list
        xlist = [item.x for item in datalist]
        cylist = [item.labels for item in datalist]
        dylist = [item.dlabels for item in datalist]
        pcylist = [item.pclabels for item in datalist]
        pdylist = [item.pdlabels for item in datalist]
        
        # Inherit properties from first dataset
        self.dataset = datalist[0].dataset
        self.task = datalist[0].task
        self.transform = datalist[0].transform
        self.target_transform = datalist[0].target_transform
        
        # Stack features and labels
        self.x = torch.vstack(xlist)
        self.labels = np.hstack(cylist)
        self.dlabels = np.hstack(dylist)
        self.pclabels = np.hstack(pcylist) if pcylist[0] is not None else None
        self.pdlabels = np.hstack(pdylist) if pdylist[0] is not None else None
