from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.nn import GCNConv, GATConv
from sklearn.cluster import KMeans

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

class GNNModel(nn.Module):
    """GNN model for activity recognition"""
    def __init__(self, input_dim, hidden_dim, num_classes, gnn_type='gcn'):
        super().__init__()
        self.gnn_type = gnn_type
        
        if gnn_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4)
            self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=1)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = torch.stack([x[batch == i].mean(0) for i in torch.unique(batch)])
        
        return self.classifier(x)

def init_gnn_model(args, input_dim, num_classes):
    """Initialize GNN model based on configuration"""
    return GNNModel(
        input_dim=input_dim,
        hidden_dim=args.gnn_hidden_dim,
        num_classes=num_classes,
        gnn_type=args.gnn_arch
    )

class Diversify(Algorithm):
    def __init__(self, args):
        super().__init__(args)
        # Feature extractor
        self.featurizer = get_fea(args)
        # Domain characterization components
        self.dbottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.domain_num)
        self.dclassifier = common_network.feat_classifier(
            args.num_classes,
            args.bottleneck,
            args.classifier
        )
        # Main classification components
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        # Auxiliary classification components
        self.abottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(
            int(args.num_classes * args.latent_domain_num),
            args.bottleneck,
            args.classifier
        )
        # Domain discrimination components
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        # Add flag for explainability mode
        self.explain_mode = False
        
        # Apply permanent fix to skip connection layer
        self.patch_skip_connection()

    def patch_skip_connection(self):
        """Permanently fix skip connection layer dimensions"""
        # Check if skip_conn exists directly
        if hasattr(self.featurizer, 'skip_conn') and isinstance(self.featurizer.skip_conn, nn.Linear):
            if self.featurizer.skip_conn.in_features != 200:
                original_out_features = self.featurizer.skip_conn.out_features
                original_device = next(self.featurizer.skip_conn.parameters()).device
                self.featurizer.skip_conn = nn.Linear(200, original_out_features)
                self.featurizer.skip_conn.to(original_device)
                print(f"Permanently patched skip_conn: in_features=200, out_features={original_out_features}")
            return
        
        # Recursively search for skip_conn in nested modules
        for name, module in self.featurizer.named_children():
            if isinstance(module, nn.Linear) and "skip" in name.lower():
                if module.in_features != 200:
                    original_out_features = module.out_features
                    original_device = next(module.parameters()).device
                    new_layer = nn.Linear(200, original_out_features)
                    new_layer.to(original_device)
                    setattr(self.featurizer, name, new_layer)
                    print(f"Permanently patched {name}: in_features=200, out_features={original_out_features}")
                return
        
        # If we reach here, no skip connection was found
        print("Warning: No skip connection layer found in featurizer")

    def update_d(self, minibatch, opt):
        """Update domain characterization components"""
        all_x1 = minibatch[0].cuda().float()
        all_d1 = minibatch[2].cuda().long()
        all_c1 = minibatch[1].cuda().long()
        
        # Validate domain labels
        n_domains = self.args.domain_num
        min_label = torch.min(all_d1).item()
        max_label = torch.max(all_d1).item()
        
        if min_label < 0 or max_label >= n_domains:
            print(f"⚠️ Domain labels out of bounds! Clamping to [0, {n_domains-1}]")
            all_d1 = torch.clamp(all_d1, 0, n_domains-1)
        
        # Forward pass
        z1 = self.dbottleneck(self.featurizer(all_x1))
        if self.explain_mode:
            z1 = z1.clone()
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        cd1 = self.dclassifier(z1)
        
        # Loss calculation
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')
        ent_loss = Entropylogits(cd1) * self.args.lam + F.cross_entropy(cd1, all_c1)
        loss = ent_loss + disc_loss
        
        # Optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        """Set pseudo-domain labels using clustering"""
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()
    
        all_fea = []
        all_index = []  # Store original indices
    
        with torch.no_grad():
            # Manually track the index counter
            index_counter = 0
            for batch in loader:
                inputs = batch[0].cuda().float()
                # Apply temporary dimension fix if needed
                inputs = self.ensure_correct_dimensions(inputs)
                feas = self.dbottleneck(self.featurizer(inputs))
                
                all_fea.append(feas.float().cpu())
                
                # Store batch indices
                batch_size = inputs.size(0)
                batch_indices = np.arange(index_counter, index_counter + batch_size)
                all_index.append(batch_indices)
                index_counter += batch_size
    
        # Combine features and indices
        all_fea = torch.cat(all_fea, dim=0)
        all_index = np.concatenate(all_index, axis=0)
        
        # Normalize features
        all_fea = all_fea / torch.norm(all_fea, p=2, dim=1, keepdim=True)
        all_fea = all_fea.float().cpu().numpy()
    
        # Clustering for pseudo-domain labels
        K = self.args.latent_domain_num
        
        # Use sklearn KMeans for robust clustering
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        pred_label = kmeans.fit_predict(all_fea)
        
        # Ensure labels are in valid range
        pred_label = np.clip(pred_label, 0, K-1)
    
        # Handle dataset types
        dataset = loader.dataset
        
        # Function to find base dataset
        def get_base_dataset(ds):
            while isinstance(ds, Subset):
                ds = ds.dataset
            return ds
        
        # Get the base dataset
        base_dataset = get_base_dataset(dataset)
        
        # Map loader indices to original dataset indices
        if isinstance(dataset, Subset):
            # Traverse through subset wrappers
            current = dataset
            while isinstance(current, Subset):
                # Map indices through each subset layer
                all_index = [current.indices[i] for i in all_index]
                current = current.dataset
            base_dataset = current
        
        # Set labels on the base dataset
        if hasattr(base_dataset, 'set_labels_by_index'):
            # Convert numpy labels to torch tensor
            pred_label_tensor = torch.from_numpy(pred_label).long()
            base_dataset.set_labels_by_index(pred_label_tensor, all_index, 'pdlabel')
            print(f"Set pseudo-labels on base dataset of type: {type(base_dataset).__name__}")
        else:
            print(f"Warning: Base dataset {type(base_dataset).__name__} has no set_labels_by_index method")
        
        # Print label distribution
        counter = Counter(pred_label)
        print(f"Pseudo-domain label distribution: {dict(counter)}")
        
        # Return to training mode
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()
    
    def ensure_correct_dimensions(self, inputs):
        """Ensure inputs have correct dimensions for skip connection"""
        # Check if input dimensions match expected skip connection dimensions
        if inputs.shape[-1] != 200:
            # Reshape to expected dimensions (batch_size, seq_len, features)
            if inputs.dim() == 2:
                # Assuming [batch_size, features]
                inputs = inputs.view(inputs.size(0), 1, inputs.size(1))
            elif inputs.dim() == 3 and inputs.size(1) == 1:
                # [batch_size, 1, features] - expand time dimension
                inputs = inputs.expand(-1, 200, -1)
            elif inputs.dim() == 3:
                # [batch_size, time, features] - transpose to match expected
                inputs = inputs.transpose(1, 2)
        return inputs

    def update(self, data, opt):
        """Update domain-invariant features"""
        all_x = data[0].cuda().float()
        all_x = self.ensure_correct_dimensions(all_x)
        all_y = data[1].cuda().long()
        all_z = self.bottleneck(self.featurizer(all_x))
        if self.explain_mode:
            all_z = all_z.clone()
        
        # Domain discrimination
        disc_input = Adver_network.ReverseLayerF.apply(all_z, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        
        # Get pseudo-domain labels directly from batch
        disc_labels = data[4].cuda().long()
        
        # Validate labels and clamp to valid range
        n_domains = self.args.latent_domain_num
        disc_labels = torch.clamp(disc_labels, 0, n_domains-1)
        
        # Ensure discriminator output size matches domain count
        if disc_out.size(1) != n_domains:
            print(f"⚠️ Discriminator output size mismatch! Expected {n_domains}, got {disc_out.size(1)}")
            # Create new discriminator with correct output size
            self.discriminator = Adver_network.Discriminator(
                self.args.bottleneck, 
                self.args.dis_hidden, 
                n_domains
            ).cuda()
            disc_out = self.discriminator(disc_input)
            
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        
        # Classification
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        
        # Combined loss
        loss = classifier_loss + disc_loss
        
        # Optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        """Update auxiliary classifier"""
        all_x = minibatches[0].cuda().float()
        all_x = self.ensure_correct_dimensions(all_x)
        all_c = minibatches[1].cuda().long()
        
        # Get pseudo-domain labels directly from batch
        all_d = minibatches[4].cuda().long()
            
        # Validate domain labels and clamp to valid range
        n_domains = self.args.latent_domain_num
        all_d = torch.clamp(all_d, 0, n_domains-1)
            
        # Create combined labels
        all_y = all_d * self.args.num_classes + all_c
        
        # Ensure combined labels are within valid range
        max_class = self.aclassifier.fc.out_features
        all_y = torch.clamp(all_y, 0, max_class-1)
        
        # Forward pass
        all_z = self.abottleneck(self.featurizer(all_x))
        if self.explain_mode:
            all_z = all_z.clone()
        all_preds = self.aclassifier(all_z)
        
        # Loss calculation and optimization
        classifier_loss = F.cross_entropy(all_preds, all_y)
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        
        return {'class': classifier_loss.item()}

    def predict(self, x):
        """Main prediction method"""
        x = self.ensure_correct_dimensions(x)
        features = self.featurizer(x)
        bottleneck_out = self.bottleneck(features)
        if self.explain_mode:
            bottleneck_out = bottleneck_out.clone()
        return self.classifier(bottleneck_out)
    
    def predict1(self, x):
        """Domain discriminator prediction"""
        x = self.ensure_correct_dimensions(x)
        features = self.featurizer(x)
        bottleneck_out = self.dbottleneck(features)
        if self.explain_mode:
            bottleneck_out = bottleneck_out.clone()
        return self.ddiscriminator(bottleneck_out)
    
    def forward(self, batch):
        inputs = batch[0]
        inputs = self.ensure_correct_dimensions(inputs)
        labels = batch[1]
        
        preds = self.predict(inputs)
        preds = preds.float()
        labels = labels.long()
        
        class_loss = self.criterion(preds, labels)
        
        return {'class': class_loss}
    
    def explain(self, x):
        original_mode = self.explain_mode
        try:
            self.explain_mode = True
            with torch.no_grad():
                x = self.ensure_correct_dimensions(x)
                return self.predict(x)
        finally:
            self.explain_mode = original_mode
