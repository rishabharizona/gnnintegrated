from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from sklearn.cluster import KMeans

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits
from torch_geometric.utils import to_dense_batch

GNN_AVAILABLE = True  # Flag indicating GNN availability
        
def transform_for_gnn(x):
    """Robust transformation for GNN input handling various formats"""
    if not GNN_AVAILABLE:
        return x  
    # Handle PyG Data objects directly
    if isinstance(x, (Data, Batch)):
        # Convert to dense representation
        x_dense, mask = to_dense_batch(x.x, x.batch)
        return x_dense
    
    # Handle common 4D formats
    if x.dim() == 4:
        # Format 1: [batch, channels, 1, time] -> [batch, time, channels]
        if x.size(1) == 8 or x.size(1) == 200:
            return x.squeeze(2).permute(0, 2, 1)
        # Format 2: [batch, 1, channels, time] -> [batch, time, channels]
        elif x.size(2) == 8 or x.size(2) == 200:
            return x.squeeze(1).permute(0, 2, 1)
        # Format 3: [batch, time, 1, channels] -> [batch, time, channels]
        elif x.size(3) == 8 or x.size(3) == 200:
            return x.squeeze(2)
        # New format: [batch, time, channels, 1]
        elif x.size(3) == 1 and (x.size(2) == 8 or x.size(2) == 200):
            return x.squeeze(3)
    
    # Handle 3D formats
    elif x.dim() == 3:
        # Format 1: [batch, channels, time] -> [batch, time, channels]
        if x.size(1) == 8 or x.size(1) == 200:
            return x.permute(0, 2, 1)
        # Format 2: [batch, time, channels] - already correct
        elif x.size(2) == 8 or x.size(2) == 200:
            return x
    
    # Unsupported format
    raise ValueError(
        f"Cannot transform input of shape {x.shape if hasattr(x, 'shape') else type(x)} for GNN. "
        f"Expected formats: PyG Data object, [B, C, 1, T], [B, 1, C, T], [B, T, 1, C], [B, T, C, 1], "
        f"or 3D formats [B, C, T] or [B, T, C] where C is 8 or 200."
    )

# Focal Loss with class balancing
class BalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights  # Should be [n_classes] tensor

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            focal_loss = weights * focal_loss
            
        return focal_loss.mean()
        
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
            nn.Dropout(0.5),
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
        self.featurizer = get_fea(args)
        self.dbottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.domain_num)
        self.dclassifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)
        self.bottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)
        self.abottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(int(args.num_classes * args.latent_domain_num), args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.args = args
        
        # Initialize with class-balanced focal loss if weights available
        if hasattr(args, 'class_weights'):
            self.criterion = BalancedFocalLoss(gamma=2.0, class_weights=args.class_weights)
        else:
            self.criterion = BalancedFocalLoss(gamma=2.0)
            
        self.lambda_cls = getattr(args, "lambda_cls", 1.0)
        self.lambda_dis = getattr(args, "lambda_dis", 0.1)
        self.explain_mode = False
        self.global_step = 0
        self.patch_skip_connection()
        
        # Initialize learning rate warmup scheduler
        self.warmup_steps = getattr(args, "warmup_steps", 1000)
        self.warmup_scheduler = None
        
    def init_optimizers(self, optimizer):
        """Initialize warmup scheduler after optimizer is created"""
        if self.warmup_steps > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda step: min(1.0, (step + 1) / self.warmup_steps)
            )
        
    def patch_skip_connection(self):
        """Dynamically adjust skip connection based on actual input shape"""
        # Get device from model parameters
        device = next(self.featurizer.parameters()).device
        
        # Create sample input on the same device as the model
        sample_input = torch.randn(1, *self.args.input_shape).to(device)
        
        # Forward pass to get actual feature dimension
        with torch.no_grad():
            actual_features = self.featurizer(sample_input).shape[-1]
            print(f"Detected actual feature dimension: {actual_features}")
        
        # Recursively search for skip_conn in nested modules
        for name, module in self.featurizer.named_modules():
            if isinstance(module, nn.Linear) and "skip" in name.lower():
                if module.in_features != actual_features:
                    print(f"Patching skip connection: {actual_features} features")
                    
                    # Create new layer with correct dimensions
                    new_layer = nn.Sequential(
                        nn.BatchNorm1d(actual_features),
                        nn.Linear(actual_features, module.out_features)
                    ).to(device)
                    
                    # Try to partially initialize with existing weights
                    if hasattr(module, 'weight'):
                        # Only try if it's a standard Linear layer
                        if module.in_features > actual_features:
                            # Use first 'actual_features' channels from existing weights
                            new_layer[1].weight.data = module.weight.data[:, :actual_features].clone()
                            new_layer[1].bias.data = module.bias.data.clone()
                        elif actual_features > module.in_features:
                            # Initialize new channels with small random values
                            new_weights = torch.randn(module.out_features, actual_features).to(device) * 0.01
                            new_weights[:, :module.in_features] = module.weight.data.clone()
                            new_layer[1].weight.data = new_weights
                            new_layer[1].bias.data = module.bias.data.clone()
                    
                    # Replace the module
                    if '.' in name:
                        # Handle nested modules (e.g., 'module.submodule.layer')
                        parts = name.split('.')
                        parent = self.featurizer
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], new_layer)
                    else:
                        setattr(self.featurizer, name, new_layer)
                    
                    print(f"Patched {name}: in_features={actual_features}, "
                          f"out_features={module.out_features}")
                return
        
        # If we reach here, no skip connection was found
        print("Warning: No skip connection layer found in featurizer")
        self.actual_features = actual_features  # Store for later use

    def update_d(self, minibatch, opt):
        """Update domain discriminator and classifier"""
        # Handle PyG Data objects
        if isinstance(minibatch[0], (Data, Batch)):
            data = minibatch[0]
            data = data.to('cuda')
            all_x1 = data
            all_c1 = data.y.long()  # Ensure integer type
            all_d1 = data.domain.long() if hasattr(data, 'domain') else minibatch[2].cuda().long()
        else:
            all_x1 = minibatch[0].cuda().float()
            all_c1 = minibatch[1].cuda().long()
            all_d1 = minibatch[2].cuda().long()
        
        # Ensure domain labels are integers
        if all_d1.dtype != torch.long:
            all_d1 = all_d1.long()
        
        n_domains = self.args.domain_num
        all_d1 = torch.clamp(all_d1, 0, n_domains - 1)
        
        # Apply data augmentation during training
        if self.training and getattr(self.args, 'use_augmentation', False):
            all_x1 = self.spectral_augmentation(all_x1)
        
        # Ensure correct dimensions for non-PyG inputs
        if not isinstance(all_x1, (Data, Batch)):
            all_x1 = self.ensure_correct_dimensions(all_x1)
        
        z1 = self.dbottleneck(self.featurizer(all_x1))
        if self.explain_mode:
            z1 = z1.clone()
            
        # Domain discrimination
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        
        # Classification
        cd1 = self.dclassifier(z1)
        
        # Loss calculations
        disc_loss = F.cross_entropy(disc_out1, all_d1)
        ent_loss = Entropylogits(cd1) * self.args.lam + self.criterion(cd1, all_c1)
        loss = ent_loss + self.lambda_dis * disc_loss
        
        # Optimization
        opt.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        
        # Debugging information
        if self.global_step % 50 == 0:
            with torch.no_grad():
                preds = cd1.argmax(dim=1)
                acc = (preds == all_c1).float().mean().item()
                domain_acc = (disc_out1.argmax(dim=1) == all_d1).float().mean().item()
                print(f"[D-Step {self.global_step}] Loss: {loss.item():.4f} | "
                      f"ClassAcc: {acc:.4f} | DomainAcc: {domain_acc:.4f}")
        
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def spectral_augmentation(self, x):
        """Apply spectral augmentation to sensor data"""
        # Random frequency masking
        if np.random.rand() > 0.5 and x.dim() >= 3:
            freq_mask = torch.ones_like(x)
            f_start = np.random.randint(0, x.shape[1]//2)
            f_length = np.random.randint(1, max(1, x.shape[1]//4))
            freq_mask[:, f_start:f_start+f_length] = 0
            x = x * freq_mask.to(x.device)
        return x

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
                if self.args.use_gnn:
                    inputs = transform_for_gnn(inputs)
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
        # Use stored feature dimension if available
        if hasattr(self, 'actual_features'):
            actual_features = self.actual_features
        else:
            # Fallback: detect feature dimension
            device = next(self.featurizer.parameters()).device
            sample_input = torch.randn(1, *self.args.input_shape).to(device)
            with torch.no_grad():
                actual_features = self.featurizer(sample_input).shape[-1]
            self.actual_features = actual_features
        
        # Reshape if needed
        if inputs.dim() == 2:
            # [batch_size, features] -> [batch_size, time, features]
            inputs = inputs.view(inputs.size(0), 1, inputs.size(1))
        elif inputs.dim() == 3 and inputs.size(1) == 1:
            # [batch_size, 1, features] - expand time dimension
            inputs = inputs.expand(-1, actual_features, -1)
        elif inputs.dim() == 3 and inputs.size(2) == actual_features:
            # [batch_size, time, features] - transpose to match expected
            inputs = inputs.permute(0, 2, 1)
        return inputs

    def update(self, data, opt):
        all_x = data[0].cuda().float()
        all_x = self.ensure_correct_dimensions(all_x)
        all_y = data[1].cuda().long()
        
        # Apply augmentation during training
        if self.training and getattr(self.args, 'use_augmentation', False):
            all_x = self.spectral_augmentation(all_x)
            
        all_z = self.bottleneck(self.featurizer(all_x))
        if self.explain_mode:
            all_z = all_z.clone()
        self.global_step += 1
        
        # Learning rate warmup
        alpha = min(1.0, self.global_step / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        disc_input = Adver_network.ReverseLayerF.apply(all_z, alpha)
        disc_out = self.discriminator(disc_input)
        
        # Domain labels
        disc_labels = data[4].cuda().long()
        disc_labels = torch.clamp(disc_labels, 0, self.args.latent_domain_num - 1)
        
        # Loss calculations
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        all_preds = self.classifier(all_z)
        classifier_loss = self.criterion(all_preds, all_y)
        loss = self.lambda_cls * classifier_loss + self.lambda_dis * disc_loss
        
        # Optimization
        opt.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        
        # Update learning rate with warmup scheduler
        if self.warmup_scheduler:
            self.warmup_scheduler.step()
        
        # Debug information
        if self.global_step % 100 == 0:
            with torch.no_grad():
                preds = all_preds.argmax(dim=1)
                acc = (preds == all_y).float().mean().item()
                domain_acc = (disc_out.argmax(dim=1) == disc_labels).float().mean().item()
                print(f"[Step {self.global_step}] ClassLoss={classifier_loss.item():.4f} | "
                      f"DiscLoss={disc_loss.item():.4f} | ClassAcc: {acc:.4f} | DomainAcc: {domain_acc:.4f}")
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        """Update auxiliary classifier"""
        # Extract inputs, class labels, and domain labels
        inputs = minibatches[0]
        
        # Handle PyG Data objects
        if isinstance(inputs, (Data, Batch)):
            inputs = inputs.to('cuda')
            all_c = inputs.y.long()
            all_d = inputs.domain.long() if hasattr(inputs, 'domain') else minibatches[2].cuda().long()
        else:
            inputs = inputs.cuda().float()
            inputs = self.ensure_correct_dimensions(inputs)
            all_c = minibatches[1].cuda().long()
            all_d = minibatches[2].cuda().long()
        
        # Ensure labels are not None
        if all_c is None:
            raise RuntimeError("Class labels are None in update_a")
        if all_d is None:
            raise RuntimeError("Domain labels are None in update_a")
        
        # Validate domain labels and clamp to valid range
        n_domains = self.args.latent_domain_num
        all_d = torch.clamp(all_d, 0, n_domains-1)
        
        # Create combined labels
        all_y = all_d * self.args.num_classes + all_c
        
        # Ensure combined labels are within valid range
        max_class = self.aclassifier.fc.out_features
        all_y = torch.clamp(all_y, 0, max_class-1)
        
        # Forward pass
        all_z = self.abottleneck(self.featurizer(inputs))
        
        if self.explain_mode:
            all_z = all_z.clone()
        
        all_preds = self.aclassifier(all_z)
        
        # Loss calculation and optimization
        classifier_loss = F.cross_entropy(all_preds, all_y)
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        
        # Debug information
        if self.global_step % 100 == 0:
            with torch.no_grad():
                preds = all_preds.argmax(dim=1)
                acc = (preds == all_y).float().mean().item()
                print(f"[Aux-Step {self.global_step}] AuxLoss={classifier_loss.item():.4f} | AuxAcc: {acc:.4f}")
        
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
