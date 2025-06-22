import torch




import torch.nn as nn


from torch_geometric.nn import GCNConv





class TemporalGCN(nn.Module):


    def __init__(self, input_dim, hidden_dim, output_dim):


        super(TemporalGCN, self).__init__()


        self.gcn1 = GCNConv(input_dim, hidden_dim)


        self.gcn2 = GCNConv(hidden_dim, hidden_dim)


        self.lstm = nn.LSTM(hidden_dim, output_dim, batch_first=True)





    def forward(self, x, edge_index, batch_size):


        x = self.gcn1(x, edge_index)


        x = torch.relu(x)Add commentMore actions


        x = self.gcn2(x, edge_index)


        x = torch.relu(x)


        x = x.view(batch_size, -1, x.size(1))


        _, (h_n, _) = self.lstm(x)


        return h_n[-1]
