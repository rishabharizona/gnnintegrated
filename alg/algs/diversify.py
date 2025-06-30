import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

class Diversify(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.featurizer = self.get_featurizer(args)
        self.classifier = nn.Linear(args.bottleneck, args.num_classes)
        self.args = args

    def get_featurizer(self, args):
        if args.use_gnn:
            return TemporalGCN(input_dim=8, hidden_dim=64, output_dim=args.bottleneck)
        else:
            return nn.Sequential(
                nn.Conv1d(8, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32, args.bottleneck)
            )
    
    def set_dlabel(self, loader):
        features = []
        with torch.no_grad():
            self.eval()
            for batch in loader:
                inputs = batch[0].to('cuda')
                feats = self.featurizer(inputs).cpu().numpy()
                features.append(feats)
        
        features = np.concatenate(features)
        kmeans = KMeans(n_clusters=self.args.latent_domain_num, n_init=10)
        pred_label = kmeans.fit_predict(features)
        loader.dataset.set_pseudo_labels(pred_label)
    
    def predict(self, x):
        features = self.featurizer(x)
        return self.classifier(features)
    
    def forward(self, x):
        return self.predict(x)

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        if isinstance(x, Batch):
            x = x.x
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return x
