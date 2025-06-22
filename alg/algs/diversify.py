from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from alg.modelopera import get_fea
from network.common_network import CommonNetwork
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

class Diversify(Algorithm):
    def __init__(self, args):
        super(Diversify, self).__init__(args)
        self.featurizer = CommonNetwork(args.dataset).cuda()
        self.classifier = nn.Linear(self.featurizer.output_num(), args.class_num).cuda()
        self.discriminator = Adver_network(args.latent_domain_num).cuda()
        self.args = args

    def predict(self, x):
        features = self.featurizer(x)
        return self.classifier(features)

    def update_a(self, data, optimizer):
        self.featurizer.train()
        self.classifier.train()
        x, y, _ = data
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        features = self.featurizer(x)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        return {'class': loss.item()}

    def update_a_mixup(self, mixed_x, y_a, y_b, lam, optimizer):
        self.featurizer.train()
        self.classifier.train()
        optimizer.zero_grad()
        features = self.featurizer(mixed_x)
        logits = self.classifier(features)
        loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
        loss.backward()
        optimizer.step()
        return {'class': loss.item()}

    def update_d(self, data, optimizer):
        self.featurizer.eval()
        self.discriminator.train()
        x, _, d = data
        x, d = x.cuda(), d.cuda()
        optimizer.zero_grad()
        features = self.featurizer(x).detach()
        logits = self.discriminator(features)
        dis_loss = F.cross_entropy(logits, d)
        ent_loss = Entropylogits(logits)
        total_loss = dis_loss + self.args.ent_weight * ent_loss
        total_loss.backward()
        optimizer.step()
        return {
            'dis': dis_loss.item(),
            'ent': ent_loss.item(),
            'total': total_loss.item()
        }

    def update(self, data, optimizer):
        self.featurizer.train()
        self.classifier.train()
        x, y, _ = data
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        features = self.featurizer(x)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        return {'class': loss.item()}

    def set_dlabel(self, loader):
        self.featurizer.eval()
        all_features, all_labels = [], []
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.cuda()
                features = self.featurizer(x)
                all_features.append(features.cpu().numpy())
        all_features = np.concatenate(all_features, axis=0)
        self.assign_domain_labels(all_features)

    def assign_domain_labels(self, features):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.args.latent_domain_num, random_state=42).fit(features)
        self.domain_labels = kmeans.labels_

    def get_parameters(self):
        return list(self.featurizer.parameters()) + list(self.classifier.parameters()) + list(self.discriminator.parameters())
