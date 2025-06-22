import torch
import numpy as np
import itertools

def build_correlation_graph(data_sample, threshold=0.3):

    """

    Build edge_index from a single EMG sample of shape (T, C).

    Args:


        data_sample: np.ndarray of shape (T, C) — time steps x channels


        threshold: float — minimum absolute correlation to form an edge

    Returns:


        edge_index: torch.LongTensor of shape [2, num_edges]


    """

    assert len(data_sample.shape) == 2, "Input must be 2D (T, C)"

    sensors = data_sample.shape[1]

    # Compute correlation matrix

    corr = np.corrcoef(data_sample.T)


    edge_index = []


    # Create edges where absolute correlation exceeds threshold


    for i, j in itertools.product(range(sensors), repeat=2):


        if i != j and abs(corr[i, j]) > threshold:


            edge_index.append([i, j])


    # Fallback: fully connected graph if correlation fails

    if len(edge_index) == 0:Add commentMore actions


        edge_index = [[i, j] for i in range(sensors) for j in range(sensors) if i != j]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index
