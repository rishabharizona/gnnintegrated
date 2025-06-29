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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits
from torch_geometric.utils import to_dense_batch

GNN_AVAILABLE = True  # Flag indicating GNN availability

# ======================= ADVANCED LOSSES =======================
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance with label smoothing"""
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Handle class imbalance
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # Label smoothing regularization
        if self.smoothing > 0:
            log_probs = F.log_softmax(inputs, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
            focal_loss += loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class EnhancedSupConLoss(nn.Module):
    """Enhanced Supervised Contrastive Loss with hard negative mining"""
    def __init__(self, temperature=0.1, margin=0.3):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Hard negative mining
        neg_mask = 1 - mask
        hardest_negatives = (similarity_matrix * neg_mask).max(dim=1, keepdim=True)[0]
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Apply margin for hard negatives
        margin_matrix = torch.zeros_like(similarity_matrix)
        margin_matrix[neg_mask.bool()] = self.margin
        log_prob -= margin_matrix
        
        # Compute mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss
# ======================= END ADVANCED LOSSES =======================

# ======================= ADVANCED MODULES =======================
class StochasticDepth(nn.Module):
    """Stochastic Depth for regularization"""
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.zeros(x.size(0), 1, 1, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch, channels, height, width]
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        return x * attn

class TemporalAttention(nn.Module):
    """Temporal Attention Module"""
    def __init__(self, seq_len):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        # Compute attention weights based on feature importance
        attn_weights = self.attn(x.mean(dim=-1))
        return x * attn_weights.unsqueeze(-1)
# ======================= END ADVANCED MODULES =======================

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
        f"Expected formats: PyG Data object, [B, C, 1, T], [B, 1, C, T], [B, T, 1, C], "
        f"or 3D formats [B, C, T] or [B, T, C] where C is 8 or 200."
    )

class Diversify(Algorithm):
    def __init__(self, args):
        if not hasattr(args, 'bottleneck_dim'):
            args.bottleneck_dim = args.bottleneck    
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
        
        # Advanced focal loss with label smoothing
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0, smoothing=0.1)
        self.explain_mode = False
        self.patch_skip_connection()
        
        # ======================= EXPONENTIAL ACCURACY ENHANCEMENTS =======================
        # Enhanced projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(args.bottleneck, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128)
        )
        self.supcon_loss = EnhancedSupConLoss(temperature=0.1, margin=0.3)
        self.contrast_weight = 0.7  # Increased weight for contrastive loss
        
        # Regularization techniques
        self.stochastic_depth = StochasticDepth(drop_prob=0.2)
        
        # Attention mechanisms (only for non-GNN models)
        if not args.use_gnn:
            # Determine input dimensions
            if args.input_shape[1] == 1:  # [C, T] format
                self.spatial_attn = SpatialAttention(args.input_shape[0])
                self.temporal_attn = TemporalAttention(args.input_shape[2])
            else:  # [T, C] format
                self.spatial_attn = SpatialAttention(args.input_shape[1])
                self.temporal_attn = TemporalAttention(args.input_shape[0])
        
        # Feature whitening for domain invariance
        self.whiten = None
        
        # Knowledge distillation components
        self.teacher_model = None
        self.distill_temp = 3.0
        self.distill_weight = 0.3
        
        # Adaptive learning rate scheduling
        self.scheduler = None
        self.cosine_scheduler = None
        # ======================= END ENHANCEMENTS =======================
    def init_whitening(self, feature_dim):
        """Initialize whitening layer with the correct feature dimension"""
        if self.whiten is None or self.whiten.num_features != feature_dim:
            self.whiten = nn.BatchNorm1d(feature_dim, affine=False)
            print(f"Initialized whitening layer for {feature_dim} features")
        return self.whiten
   
    def configure_optimizers(self, args):
        """Enhanced optimizer configuration with adaptive scheduling"""
        params = [
            {'params': self.featurizer.parameters(), 'lr': args.lr},
            {'params': self.bottleneck.parameters(), 'lr': args.lr},
            {'params': self.classifier.parameters(), 'lr': args.lr * 0.1},
            {'params': self.projection_head.parameters(), 'lr': args.lr},
            {'params': self.abottleneck.parameters(), 'lr': args.lr},
            {'params': self.aclassifier.parameters(), 'lr': args.lr * 0.1},
            {'params': self.discriminator.parameters(), 'lr': args.lr},
        ]
        
        # Add attention parameters if they exist
        if hasattr(self, 'spatial_attn'):
            params.append({'params': self.spatial_attn.parameters(), 'lr': args.lr})
        if hasattr(self, 'temporal_attn'):
            params.append({'params': self.temporal_attn.parameters(), 'lr': args.lr})
        
        optimizer = torch.optim.AdamW(
            params, 
            lr=args.lr,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-4
        )
        
        # Adaptive learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Cosine annealing for fine-tuning
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.steps if hasattr(args, 'steps') else 100,
            eta_min=args.lr * 0.01
        )
        
        return optimizer

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
                    
                    # Replace the module
                    if '.' in name:
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
        self.actual_features = actual_features

    def update_d(self, minibatch, opt):
        """Update domain discriminator and classifier"""
        # Handle PyG Data objects
        if isinstance(minibatch[0], (Data, Batch)):
            data = minibatch[0]
            data = data.to('cuda')
            all_x1 = data
            
            # Get class labels from Data object or fallback
            if hasattr(data, 'y') and data.y is not None:
                all_c1 = data.y
            else:
                all_c1 = minibatch[1].cuda().long()
            
            # Get domain labels from Data object or fallback
            if hasattr(data, 'domain') and data.domain is not None:
                all_d1 = data.domain
            else:
                all_d1 = minibatch[2].cuda().long()
        else:
            all_x1 = minibatch[0].cuda().float()
            all_c1 = minibatch[1].cuda().long()
            all_d1 = minibatch[2].cuda().long()
        
        # Ensure proper tensor types
        all_c1 = all_c1.long() if not isinstance(all_c1, torch.Tensor) else all_c1.long()
        all_d1 = all_d1.long() if not isinstance(all_d1, torch.Tensor) else all_d1.long()
        
        n_domains = self.args.domain_num
        all_d1 = torch.clamp(all_d1, 0, n_domains - 1)
        
        # Apply GNN transformation if needed
        if self.args.use_gnn:
            all_x1 = transform_for_gnn(all_x1)
            
        # Ensure correct dimensions for non-PyG inputs
        if not isinstance(all_x1, (Data, Batch)):
            all_x1 = self.ensure_correct_dimensions(all_x1)
        
        # Apply attention if not using GNN
        if not self.args.use_gnn:
            if all_x1.dim() == 3:
                all_x1 = all_x1.permute(0, 2, 1).unsqueeze(2)
            all_x1 = self.spatial_attn(all_x1)
            all_x1 = self.temporal_attn(all_x1.squeeze(2).permute(0, 2, 1))
            all_x1 = all_x1.unsqueeze(2)
        
        z1 = self.dbottleneck(self.featurizer(all_x1))
        if self.explain_mode:
            z1 = z1.clone()
            
        # Domain discrimination
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        
        # Classification
        cd1 = self.dclassifier(z1)
        
        # Loss calculations - reduced domain loss weight
        disc_loss = F.cross_entropy(disc_out1, all_d1)
        ent_loss = Entropylogits(cd1) * self.args.lam + self.criterion(cd1, all_c1)
        loss = ent_loss + 0.01 * disc_loss  # Reduced domain loss weight
        
        # Check for NaN/inf loss
        if not torch.isfinite(loss):
            print(f"Warning: non-finite loss detected in update_d: {loss.item()}. Skipping step.")
            return {
                'total': 0,
                'dis': 0,
                'ent': 0
            }
        
        # Optimization
        opt.zero_grad()
        loss.backward()
        
        # Gradient clipping with increased max_norm
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Reduced max norm
        opt.step()
        
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        """Set pseudo-domain labels using clustering with proper PyG Data handling"""
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()
    
        all_fea = []
        all_index = []  # Store original indices
    
        with torch.no_grad():
            index_counter = 0
            for batch in loader:
                if isinstance(batch[0], (Data, Batch)):
                    data = batch[0].to('cuda')
                    inputs = data
                else:
                    inputs = batch[0].cuda().float()
                
                # Apply GNN transformation if needed
                if self.args.use_gnn:
                    inputs = transform_for_gnn(inputs)
                
                if not isinstance(inputs, (Data, Batch)):
                    inputs = self.ensure_correct_dimensions(inputs)
                
                # Apply attention if not using GNN
                if not self.args.use_gnn:
                    if inputs.dim() == 3:
                        inputs = inputs.permute(0, 2, 1).unsqueeze(2)
                    inputs = self.spatial_attn(inputs)
                    inputs = self.temporal_attn(inputs.squeeze(2).permute(0, 2, 1))
                    inputs = inputs.unsqueeze(2)
                
                feas = self.dbottleneck(self.featurizer(inputs))
                
                # Check for NaN in features
                if torch.isnan(feas).any():
                    print("Warning: NaN detected in features. Replacing with zeros.")
                    feas = torch.nan_to_num(feas, nan=0.0)
                
                all_fea.append(feas.float().cpu())
                
                batch_size = inputs.size(0) if not isinstance(inputs, (Data, Batch)) else inputs.num_graphs
                batch_indices = np.arange(index_counter, index_counter + batch_size)
                all_index.append(batch_indices)
                index_counter += batch_size
    
        # Combine features and indices
        all_fea = torch.cat(all_fea, dim=0)
        all_index = np.concatenate(all_index, axis=0)
        
        # Convert to numpy and check for NaN
        all_fea_np = all_fea.float().cpu().numpy()
        nan_mask = np.isnan(all_fea_np).any(axis=1)
        if np.any(nan_mask):
            print(f"Warning: Found {nan_mask.sum()} samples with NaN features. Replacing with 0.")
            all_fea_np[nan_mask] = 0
        
        # Normalize features safely
        norms = np.linalg.norm(all_fea_np, axis=1, keepdims=True)
        zero_norms = (norms == 0).flatten()  # Flatten to 1D for indexing
        norms[zero_norms.reshape(-1,1)] = 1  # Avoid division by zero
        
        all_fea_norm = all_fea_np / norms
        all_fea_norm[zero_norms] = 0  # Set zero vectors to zero
        
        # Feature whitening before clustering
        all_fea_norm = (all_fea_norm - all_fea_norm.mean(0)) / (all_fea_norm.std(0) + 1e-8)
        
        # Clustering for pseudo-domain labels
        K = self.args.latent_domain_num
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        pred_label = kmeans.fit_predict(all_fea_norm)
        pred_label = np.clip(pred_label, 0, K-1)
    
        # Handle dataset types
        dataset = loader.dataset
        
        # Function to find base dataset
        def get_base_dataset(ds):
            while isinstance(ds, Subset):
                ds = ds.dataset
            return ds
        
        base_dataset = get_base_dataset(dataset)
        
        if isinstance(dataset, Subset):
            current = dataset
            while isinstance(current, Subset):
                all_index = [current.indices[i] for i in all_index]
                current = current.dataset
            base_dataset = current
        
        # Set labels on the base dataset
        if hasattr(base_dataset, 'set_labels_by_index'):
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
        if hasattr(self, 'actual_features'):
            actual_features = self.actual_features
        else:
            device = next(self.featurizer.parameters()).device
            sample_input = torch.randn(1, *self.args.input_shape).to(device)
            with torch.no_grad():
                actual_features = self.featurizer(sample_input).shape[-1]
            self.actual_features = actual_features
        
        # Reshape if needed
        if inputs.dim() == 2:
            inputs = inputs.view(inputs.size(0), 1, inputs.size(1))
        elif inputs.dim() == 3 and inputs.size(1) == 1:
            inputs = inputs.expand(-1, actual_features, -1)
        elif inputs.dim() == 3 and inputs.size(2) == actual_features:
            inputs = inputs.permute(0, 2, 1)
        return inputs

    def update(self, data, opt):
        """Main update method with PyG data support and exponential enhancements"""
        # Handle PyG Data objects
        if isinstance(data[0], (Data, Batch)):
            batch_data = data[0].to('cuda')
            all_x = batch_data
            
            # Get class labels from Data object or fallback
            if hasattr(batch_data, 'y') and batch_data.y is not None:
                all_y = batch_data.y
            else:
                all_y = data[1].cuda().long()
                
            # Get domain labels from Data object or fallback
            if hasattr(batch_data, 'domain') and batch_data.domain is not None:
                disc_labels = batch_data.domain
            else:
                disc_labels = data[1].cuda().long()
        else:
            all_x = data[0].cuda().float()
            all_y = data[1].cuda().long()
            disc_labels = data[4].cuda().long()
        
        # Apply GNN transformation if needed
        if self.args.use_gnn:
            all_x = transform_for_gnn(all_x)
            
        # Ensure correct dimensions for non-PyG inputs
        if not isinstance(all_x, (Data, Batch)):
            all_x = self.ensure_correct_dimensions(all_x)
        
        # Apply attention if not using GNN
        if not self.args.use_gnn:
            if all_x.dim() == 3:
                all_x = all_x.permute(0, 2, 1).unsqueeze(2)
            all_x = self.spatial_attn(all_x)
            all_x = self.temporal_attn(all_x.squeeze(2).permute(0, 2, 1))
            all_x = all_x.unsqueeze(2)
        
        # Forward pass with stochastic depth
        features = self.featurizer(all_x)
        features = self.stochastic_depth(features)
        all_z = self.bottleneck(features)
        # Initialize whitening if needed
        if self.whiten is None:
            self.init_whitening(all_z.size(1))
        # Feature whitening for domain invariance
        all_z = self.whiten(all_z)
            
        # ======================= CONTRASTIVE LEARNING =======================
        # Project features for contrastive loss
        projections = self.projection_head(all_z)
        contrastive_loss = self.supcon_loss(projections, all_y) * self.contrast_weight
        # ======================= END CONTRASTIVE LEARNING =======================
            
        # Domain discrimination
        disc_input = Adver_network.ReverseLayerF.apply(all_z, 1.0)
        disc_out = self.discriminator(disc_input)
        
        # Domain labels
        disc_labels = torch.clamp(disc_labels, 0, self.args.latent_domain_num - 1)
        
        # Loss calculations - reduced domain loss weight
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        all_preds = self.classifier(all_z)
        classifier_loss = self.criterion(all_preds, all_y)
        
        # ======================= KNOWLEDGE DISTILLATION =======================
        # Teacher-student distillation if available
        distill_loss = 0
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model.predict(all_x)
            
            teacher_probs = F.softmax(teacher_logits / self.distill_temp, dim=-1)
            student_logits = all_preds / self.distill_temp
            
            distill_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction='batchmean'
            ) * (self.distill_temp ** 2)
        # ======================= END DISTILLATION =======================
        
        # Combined loss
        loss = (
            classifier_loss + 
            0.01 * disc_loss + 
            contrastive_loss +
            self.distill_weight * distill_loss
        )
        
        # Check for NaN/inf loss
        if not torch.isfinite(loss):
            print(f"Warning: non-finite loss detected: {loss.item()}. Skipping step.")
            return {
                'total': 0,
                'class': classifier_loss.item() if torch.isfinite(classifier_loss) else 0,
                'dis': disc_loss.item() if torch.isfinite(disc_loss) else 0,
                'contrast': contrastive_loss.item() if torch.isfinite(contrastive_loss) else 0,
                'distill': distill_loss.item() if torch.isfinite(distill_loss) else 0
            }
        
        # Optimization
        opt.zero_grad()
        loss.backward()
        
        # Adaptive gradient clipping
        max_norm = 1.0 + 0.1 * (1 - opt.param_groups[0]['lr'] / self.args.lr)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
        opt.step()
        
        # Update learning rate
        if self.cosine_scheduler:
            self.cosine_scheduler.step()
        
        return {
            'total': loss.item(), 
            'class': classifier_loss.item(), 
            'dis': disc_loss.item(),
            'contrast': contrastive_loss.item(),
            'distill': distill_loss.item() if self.teacher_model else 0
        }

    def update_a(self, minibatches, opt):
        """Update auxiliary classifier with robust label handling"""
        # Extract inputs, class labels, and domain labels
        inputs = minibatches[0]
        
        # Handle PyG Data objects
        if isinstance(inputs, (Data, Batch)):
            inputs = inputs.to('cuda')
            
            if hasattr(inputs, 'y') and inputs.y is not None:
                all_c = inputs.y
            else:
                if len(minibatches) > 1:
                    all_c = minibatches[1]
                else:
                    raise RuntimeError("Class labels not found")
            
            if hasattr(inputs, 'domain') and inputs.domain is not None:
                all_d = inputs.domain
            else:
                if len(minibatches) > 2:
                    all_d = minibatches[2]
                else:
                    raise RuntimeError("Domain labels not found")
        else:
            inputs = inputs.cuda().float()
            all_c = minibatches[1]
            all_d = minibatches[2]
        
        # Convert to tensors and ensure proper types
        if not isinstance(all_c, torch.Tensor):
            all_c = torch.tensor(all_c, device='cuda')
        all_c = all_c.long()
        
        if not isinstance(all_d, torch.Tensor):
            all_d = torch.tensor(all_d, device='cuda')
        all_d = all_d.long()
        
        # Validate domain labels and clamp to valid range
        n_domains = self.args.latent_domain_num
        all_d = torch.clamp(all_d, 0, n_domains-1)
        
        # Create combined labels
        all_y = all_d * self.args.num_classes + all_c
        
        # Ensure combined labels are within valid range
        max_class = self.aclassifier.fc.out_features
        if all_y.max() >= max_class:
            all_y = torch.clamp(all_y, 0, max_class-1)
        
        # Apply GNN transformation if needed
        if self.args.use_gnn:
            inputs = transform_for_gnn(inputs)
            
        # Ensure correct dimensions for non-PyG inputs
        if not isinstance(inputs, (Data, Batch)):
            inputs = self.ensure_correct_dimensions(inputs)
        
        # Apply attention if not using GNN
        if not self.args.use_gnn:
            if inputs.dim() == 3:
                inputs = inputs.permute(0, 2, 1).unsqueeze(2)
            inputs = self.spatial_attn(inputs)
            inputs = self.temporal_attn(inputs.squeeze(2).permute(0, 2, 1))
            inputs = inputs.unsqueeze(2)
        
        # Forward pass
        all_z = self.abottleneck(self.featurizer(inputs))
        all_preds = self.aclassifier(all_z)
        
        # Loss calculation and optimization
        classifier_loss = F.cross_entropy(all_preds, all_y)
        
        # Check for NaN/inf loss
        if not torch.isfinite(classifier_loss):
            print(f"Warning: non-finite loss detected in update_a: {classifier_loss.item()}. Skipping step.")
            return {'class': 0}
        
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        
        return {'class': classifier_loss.item()}

    # In alg/algs/diversify.py, modify the predict method
    def predict(self, x):
        """Enhanced prediction with attention and feature whitening"""
        if not isinstance(x, (Data, Batch)):
            x = self.ensure_correct_dimensions(x)
        
        # Apply attention if not using GNN
        if not self.args.use_gnn:
            if x.dim() == 3:
                x = x.permute(0, 2, 1).unsqueeze(2)
            x = self.spatial_attn(x)
            x = self.temporal_attn(x.squeeze(2).permute(0, 2, 1))
            x = x.unsqueeze(2)
        
        features = self.featurizer(x)
        bottleneck_out = self.bottleneck(features)
        
        # Initialize whitening if needed
        if self.whiten is None:
            # Create whitening layer with correct dimension
            feature_dim = bottleneck_out.size(1)
            self.whiten = nn.BatchNorm1d(feature_dim, affine=False).to(bottleneck_out.device)
            print(f"Initialized whitening layer for {feature_dim} features during prediction")
        
        # Handle small batches during curriculum phase
        if bottleneck_out.size(0) == 1:
            # Use instance normalization for single-sample batches
            whitened = (bottleneck_out - bottleneck_out.mean(dim=1, keepdim=True)) / (
                bottleneck_out.std(dim=1, keepdim=True) + 1e-5)
        else:
            # Use batch norm for larger batches
            whitened = self.whiten(bottleneck_out)
        
        return self.classifier(whitened)
    
    def forward(self, batch):
        inputs = batch[0]
        if not isinstance(inputs, (Data, Batch)):
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
                if not isinstance(x, (Data, Batch)):
                    x = self.ensure_correct_dimensions(x)
                return self.predict(x)
        finally:
            self.explain_mode = original_mode
            
    def update_teacher(self, exponential_moving_average=0.999):
        """EMA update for teacher model"""
        if self.teacher_model is None:
            # Initialize teacher with current weights
            self.teacher_model = self.__class__(self.args)
            self.teacher_model.load_state_dict(self.state_dict())
            self.teacher_model.eval()
            return
        
        # Update teacher with EMA
        teacher_params = dict(self.teacher_model.named_parameters())
        student_params = dict(self.named_parameters())
        
        for name in student_params:
            if name in teacher_params:
                teacher_params[name].data.mul_(exponential_moving_average).add_(
                    student_params[name].data, alpha=1 - exponential_moving_average
                )
