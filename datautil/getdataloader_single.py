import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people
from torch_geometric.data import Batch, Data
from datautil.graph_utils import convert_to_graph

# Task mapping for activity recognition
task_act = {'cross_people': cross_people}

class SafeSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.indices = indices

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return self.convert_data(data)

    def convert_data(self, data):
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
            for key in data.keys:
                data[key] = self.convert_data(data[key])
            return data
        else:
            try:
                return torch.tensor(data)
            except:
                return data

def collate_gnn(batch):
    graphs, labels, domains = zip(*batch)
    batched_graph = Batch.from_data_list(graphs)
    labels = torch.tensor(labels, dtype=torch.long)
    domains = torch.tensor(domains, dtype=torch.long)
    return batched_graph, labels, domains

def get_gnn_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=collate_gnn
    )

def get_dataloader(args, tr, val, tar):
    assert args.model_type == 'gnn', "Expected model_type to be 'gnn'"

    train_loader = get_gnn_dataloader(tr, args.batch_size, args.N_WORKERS, shuffle=True)
    train_loader_noshuffle = get_gnn_dataloader(tr, args.batch_size, args.N_WORKERS, shuffle=False)
    valid_loader = get_gnn_dataloader(val, args.batch_size, args.N_WORKERS, shuffle=False)
    target_loader = get_gnn_dataloader(tar, args.batch_size, args.N_WORKERS, shuffle=False)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader

def get_act_dataloader(args):
    source_datasetlist = []
    target_datalist = []
    pcross_act = task_act[args.task]

    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)

    for i, item in enumerate(tmpp):
        transform = actutil.act_to_graph_transform(args)
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=transform
        )

        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)
            if len(tdata)/args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata)/args.batch_size

    rate = 0.2
    args.steps_per_epoch = int(args.steps_per_epoch * (1 - rate))

    tdata = combindataset(args, source_datasetlist)
    l = len(tdata.labels)
    indexall = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l * rate)
    indextr, indexval = indexall[ted:], indexall[:ted]
    tr = SafeSubset(tdata, indextr)
    val = SafeSubset(tdata, indexval)

    targetdata = combindataset(args, target_datalist)
    loaders = get_dataloader(args, tr, val, targetdata)
    return (*loaders, tr, val, targetdata)

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    domain_indices = {}
    for idx in range(len(val_dataset)):
        domain = val_dataset[idx][2]
        if isinstance(domain, torch.Tensor):
            domain = domain.item()
        domain_indices.setdefault(domain, []).append(idx)

    domain_metrics = []
    with torch.no_grad():
        for domain, indices in domain_indices.items():
            subset = Subset(val_dataset, indices)
            loader = DataLoader(
                subset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS,
                collate_fn=collate_gnn
            )
            total_loss, correct, total, num_batches = 0, 0, 0, 0
            for batch in loader:
                inputs = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                output = algorithm.predict(inputs)
                loss = torch.nn.functional.cross_entropy(output, labels)
                total_loss += loss.item()
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                num_batches += 1
            avg_loss = total_loss / max(num_batches, 1)
            acc = correct / max(total, 1)
            domain_metrics.append((domain, avg_loss, acc))

    losses = [m[1] for m in domain_metrics]
    accs = [m[2] for m in domain_metrics]
    min_loss, max_loss = min(losses), max(losses)
    min_acc, max_acc = min(accs), max(accs)
    domain_scores = []
    for domain, loss, acc in domain_metrics:
        norm_loss = (loss - min_loss) / (max_loss - min_loss + 1e-8)
        norm_acc = (acc - min_acc) / (max_acc - min_acc + 1e-8)
        difficulty = 0.7 * norm_loss + 0.3 * (1 - norm_acc)
        domain_scores.append((domain, difficulty))

    domain_scores.sort(key=lambda x: x[1])
    num_domains = len(domain_scores)
    progress = min(1.0, (stage + 1) / args.CL_PHASE_EPOCHS)
    progress = np.sqrt(progress)
    num_selected = max(2, int(progress * num_domains * 0.8))
    selected_domains = [domain for domain, _ in domain_scores[:num_selected]]

    if len(domain_scores) > num_selected:
        selected_domains.append(random.choice(domain_scores[num_selected:])[0])

    train_domain_indices = {}
    for idx in range(len(train_dataset)):
        domain = train_dataset[idx][2]
        if isinstance(domain, torch.Tensor):
            domain = domain.item()
        train_domain_indices.setdefault(domain, []).append(idx)

    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            domain_indices = train_domain_indices[domain]
            n_samples = max(50, int(len(domain_indices) * 0.7))
            selected_indices.extend(random.sample(domain_indices, n_samples))

    if not selected_indices:
        selected_indices = list(range(len(train_dataset)))

    curriculum_subset = SafeSubset(train_dataset, selected_indices)
    curriculum_loader = DataLoader(
        curriculum_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.N_WORKERS,
        drop_last=True, collate_fn=collate_gnn
    )
    return curriculum_loader

def get_shap_batch(loader, size=100):
    X_val = []
    for batch in loader:
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        X_val.append(inputs)
        if len(torch.cat(X_val)) >= size:
            break
    return torch.cat(X_val)[:size]
