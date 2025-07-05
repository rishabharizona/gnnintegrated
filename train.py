import os
import sys
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ, disable_inplace_relu
from datautil.getdataloader_single import get_act_dataloader, get_curriculum_loader
from torch_geometric.loader import DataLoader as PyGDataLoader 
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader
from network.act_network import ActNetwork
import random  # Added for data augmentation
import types
from alg.algs.diversify import Diversify
# Suppress TensorFlow and SHAP warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger("shap").setLevel(logging.WARNING)

# Unified SHAP utilities import
from shap_utils import (
    get_background_batch, safe_compute_shap_values, plot_summary,
    overlay_signal_with_shap, plot_shap_heatmap,
    evaluate_shap_impact, compute_flip_rate, compute_jaccard_topk,
    compute_kendall_tau,
    cosine_similarity_shap, save_shap_numpy,
    compute_confidence_change, _get_shap_array,
    compute_aopc, compute_feature_coherence, compute_shap_entropy,
    plot_emg_shap_4d, plot_4d_shap_surface, evaluate_advanced_shap_metrics
)

# ======================= GNN INTEGRATION START =======================
try:
    from gnn.temporal_gcn import TemporalGCN
    from gnn.graph_builder import GraphBuilder
    from torch_geometric.data import Data, Batch
    GNN_AVAILABLE = True
    print("GNN modules successfully imported")
except ImportError as e:
    print(f"[WARNING] GNN modules not available: {str(e)}")
    print("Falling back to CNN architecture")
    GNN_AVAILABLE = False
    # Define dummy classes for non-GNN environments
    class Data:
        pass
    class Batch:
        pass
# ======================= GNN INTEGRATION END =======================

def automated_k_estimation(features, k_min=2, k_max=10):
    """Automatically determine optimal cluster count using silhouette score and Davies-Bouldin Index"""
    best_k = k_min
    best_score = -1
    scores = []
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features)
        labels = kmeans.labels_
        
        if len(np.unique(labels)) < 2:
            silhouette = -1
            dbi = float('inf')
            ch_score = -1
        else:
            silhouette = silhouette_score(features, labels)
            dbi = davies_bouldin_score(features, labels)
            ch_score = calinski_harabasz_score(features, labels)
        
        # Minimal parameter impact
        norm_silhouette = (silhouette + 1) / 2
        norm_dbi = 1 / (1 + dbi)
        norm_ch = ch_score / 1000
        
        combined_score = (0.5 * norm_silhouette) + (0.3 * norm_ch) + (0.2 * norm_dbi)
        scores.append((k, silhouette, dbi, ch_score, combined_score))
        
        print(f"K={k}: Silhouette={silhouette:.4f}, DBI={dbi:.4f}, CH={ch_score:.4f}, Combined={combined_score:.4f}")
        
        if combined_score > best_score:
            best_k = k
            best_score = combined_score
            
    print(f"[INFO] Optimal K determined as {best_k} (Combined Score: {best_score:.4f})")
    return best_k

def calculate_h_divergence(features_source, features_target):
    """Minimal domain discrepancy calculation"""
    labels_source = np.zeros(features_source.shape[0])
    labels_target = np.ones(features_target.shape[0])
    
    X = np.vstack([features_source, features_target])
    y = np.hstack([labels_source, labels_target])
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    domain_classifier = LogisticRegression(max_iter=500, random_state=42)
    domain_classifier.fit(X_train, y_train)
    
    domain_acc = domain_classifier.score(X_test, y_test)
    h_divergence = 2 * (1 - 2 * (1 - domain_acc))
    
    return h_divergence, domain_acc

def transform_for_gnn(x):
    if not GNN_AVAILABLE:
        return x
    
    from torch_geometric.data import Data
    from torch_geometric.utils import to_dense_batch
    
    if isinstance(x, Data):
        x_dense, mask = to_dense_batch(x.x, x.batch)
        return x_dense
    
    if x.dim() == 4:
        if x.size(1) == 8 or x.size(1) == 200:
            return x.squeeze(2).permute(0, 2, 1)
        elif x.size(2) == 8 or x.size(2) == 200:
            return x.squeeze(1).permute(0, 2, 1)
        elif x.size(3) == 8 or x.size(3) == 200:
            return x.squeeze(2)
        elif x.size(3) == 1 and (x.size(2) == 8 or x.size(2) == 200):
            return x.squeeze(3)
    
    elif x.dim() == 3:
        if x.size(1) == 8 or x.size(1) == 200:
            return x.permute(0, 2, 1)
        elif x.size(2) == 8 or x.size(2) == 200:
            return x
    
    raise ValueError(f"Cannot transform input shape {x.shape} for GNN")

# ======================= TEMPORAL CONVOLUTION BLOCK =======================
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.0):  # ZERO DROPOUT
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.activation = nn.ReLU()  # Changed from GELU to ReLU
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
        self.padding = padding

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.conv1.bias is not None:
            self.conv1.bias.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.activation(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        return self.activation(out + residual)

# ======================= DATA AUGMENTATION MODULE =======================
class EMGDataAugmentation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.jitter_scale = 0.1  # ENABLED FOR CONTRASTIVE LEARNING
        self.scaling_std = 0.1   # ENABLED FOR CONTRASTIVE LEARNING
        self.warp_ratio = 0.1    # ENABLED FOR CONTRASTIVE LEARNING
        self.aug_prob = 0.5      # ENABLED FOR CONTRASTIVE LEARNING

    def forward(self, x):
        # Only apply augmentation during training
        if not self.training:
            return x
            
        # Apply jitter
        if random.random() < self.aug_prob:
            jitter = torch.randn_like(x) * self.jitter_scale
            x = x + jitter
            
        # Apply scaling
        if random.random() < self.aug_prob:
            scale_factor = torch.randn(x.size(0), 1, 1, device=x.device) * self.scaling_std + 1.0
            x = x * scale_factor
            
        # Apply time warping
        if random.random() < self.aug_prob and x.dim() > 2:
            seq_len = x.size(2)
            warp_points = max(3, seq_len // 10)
            warp_strength = self.warp_ratio * seq_len
            
            for i in range(x.size(0)):
                # Create random warping path
                orig_points = np.linspace(0, seq_len-1, num=warp_points)
                warp_values = np.random.normal(0, warp_strength, warp_points)
                new_points = orig_points + warp_values
                new_points[0] = 0
                new_points[-1] = seq_len-1
                
                # Interpolate
                orig_idx = np.arange(seq_len)
                warped_signal = np.interp(orig_idx, new_points, orig_points)
                
                # Apply warping
                for ch in range(x.size(1)):
                    x[i, ch] = torch.from_numpy(np.interp(orig_idx, warped_signal, x[i, ch].cpu().numpy())).to(x.device)
                    
        return x
# ======================= END AUGMENTATION MODULE =======================

# ======================= OPTIMIZER FUNCTION =======================
def get_optimizer(algorithm, args, nettype='Diversify'):
    params = algorithm.parameters()
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, 
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True)
    else:  # Default to Adam
        optimizer = torch.optim.Adam(
            params, 
            lr=args.lr,
            weight_decay=args.weight_decay)
    
    return optimizer

# ======================= TEMPORAL GCN LAYER =======================
class TemporalGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, graph_builder):
        super().__init__()
        self.graph_builder = graph_builder
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()  # Changed from GELU to ReLU
        
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        
        edge_indices = self.graph_builder.build_graph_for_batch(x)
        
        outputs = []
        for i in range(batch_size):
            sample_features = x[i]
            edge_index = edge_indices[i]
            
            if edge_index.numel() > 0:
                adj_matrix = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.size(1), device=x.device),
                    size=(seq_len, seq_len)
                ).to_dense()
            else:
                adj_matrix = torch.eye(seq_len, device=x.device)
                
            conv_result = torch.mm(adj_matrix, sample_features)
            outputs.append(conv_result)
        
        x = torch.stack(outputs, dim=0)
        x = self.linear(x)
        x = self.activation(x)
        return x

# ======================= ENHANCED GNN ARCHITECTURE =======================
class EnhancedTemporalGCN(TemporalGCN):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder, 
                 n_layers=1, use_tcn=False, lstm_hidden_size=64, 
                 lstm_layers=1, bidirectional=False, num_classes=6):
        super().__init__(input_dim, hidden_dim, output_dim, graph_builder)
        
        # Store original parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_builder = graph_builder
        self.n_layers = n_layers
        self.use_tcn = use_tcn
        self.shap_mode = False  # Flag for SHAP compatibility mode
        
        # Existing components
        self.skip_conn = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(self.n_layers):
            layer = TemporalGCNLayer(
                input_dim=self.input_dim if i == 0 else self.hidden_dim,
                output_dim=self.hidden_dim,
                graph_builder=self.graph_builder
            )
            self.gnn_layers.append(layer)
            self.norms.append(nn.LayerNorm(self.hidden_dim))
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=2,
            dropout=0.0,
            batch_first=True
        )
        
        if self.use_tcn:
            tcn_layers = []
            num_channels = [self.hidden_dim] * 2
            kernel_size = 3
            for i in range(len(num_channels)):
                dilation = 2 ** i
                in_channels = self.hidden_dim if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
                tcn_layers += [TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation, dropout=0.0
                )]
            self.tcn = nn.Sequential(*tcn_layers)
            self.tcn_proj = nn.Linear(num_channels[-1], self.output_dim)
        else:
            self.lstm = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0
            )
            lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
            self.lstm_proj = nn.Linear(lstm_output_dim, self.output_dim)
            self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        
        self.temporal_norm = nn.LayerNorm(self.output_dim)
        
        # SHAP-specific components
        self.shap_projection = None
        self.shap_classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(8 * 200, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
        )
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.gnn_layers:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.norms:
            if hasattr(layer, 'weight'):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
        if hasattr(self, 'lstm'):
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                     
        # Initialize SHAP classifier weights
        for layer in self.shap_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                    

    def forward(self, x):
        """Main forward pass with SHAP mode support"""
        if self.shap_mode:
            return self.forward_shap(x)
            
        # Original forward pass implementation
        if hasattr(x, 'x') and hasattr(x, 'batch'):
            from torch_geometric.utils import to_dense_batch
            x, mask = to_dense_batch(x.x, x.batch)
        
        # Store original device
        device = x.device
        
        # Dynamic feature dimension handling
        if x.dim() == 4:  # [batch, channels, spatial, time]
            # Handle EMG data format
            if x.size(1) == 8 or x.size(1) == 200:
                x = x.squeeze(2).permute(0, 2, 1)
            elif x.size(2) == 8 or x.size(2) == 200:
                x = x.squeeze(1).permute(0, 2, 1)
            elif x.size(3) == 8 or x.size(3) == 200:
                x = x.squeeze(2)
            elif x.size(3) == 1 and (x.size(2) == 8 or x.size(2) == 200):
                x = x.squeeze(3)
        
        # Handle PyG Batch objects
        if isinstance(x, Batch):
            x = to_dense_batch(x.x, x.batch)[0]
        
        # Get actual feature dimension
        actual_dim = x.size(-1)
        
        # Dynamic projection for unexpected dimensions
        if actual_dim != self.input_dim:
            if not hasattr(self, 'dynamic_projection'):
                self.dynamic_projection = nn.Linear(actual_dim, self.input_dim).to(device)
            x = self.dynamic_projection(x)
        
        # Continue with original processing
        original_x = x.clone()
        
        if x.size(-1) == 200 and self.input_dim == 8:
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Linear(200, 8).to(x.device)
            x = self.feature_projection(x)
        
        gnn_features = x
        for layer, norm in zip(self.gnn_layers, self.norms):
            gnn_features = layer(gnn_features)
            gnn_features = norm(gnn_features)
            gnn_features = F.relu(gnn_features)
        
        attn_out, _ = self.attention(gnn_features, gnn_features, gnn_features)
        x = gnn_features + attn_out
        
        if self.use_tcn:
            tcn_in = x.permute(0, 2, 1)
            tcn_out = self.tcn(tcn_in)
            tcn_out = tcn_out.permute(0, 2, 1)
            temporal_out = self.tcn_proj(tcn_out)
        else:
            lstm_out, _ = self.lstm(x)
            lstm_out = self.lstm_norm(lstm_out)
            temporal_out = self.lstm_proj(lstm_out)
        
        gnn_out = temporal_out.mean(dim=1)
        gnn_out = self.temporal_norm(gnn_out)
        
        skip_out = gnn_features
        skip_out = skip_out.mean(dim=1)
        skip_out = self.skip_conn(skip_out)
        
        gate = torch.sigmoid(0.1 * gnn_out + 0.1 * skip_out)
        output = gate * gnn_out + (1 - gate) * skip_out
        
        return output
    
    # In EnhancedTemporalGCN.forward_shap
    def forward_shap(self, x):
        """Simplified forward pass for SHAP compatibility"""
        # Extract features from PyG objects if needed
        if hasattr(x, 'x'):
            features = x.x
        else:
            features = x
        
        
        # Handle feature dimensions - flatten properly
        if features.dim() == 2:  # [nodes, features]
            # For single graph: [8, 200] -> [1, 1600]
            features = features.flatten().unsqueeze(0)
        elif features.dim() == 3:  # [batch, nodes, features]
            # For batch: [batch, 8, 200] -> [batch, 1600]
            features = features.flatten(start_dim=1)
        elif features.dim() == 4:  # [batch, ch, spatial, time]
            features = features.flatten(start_dim=1)
        
        
        # Get actual feature dimension
        actual_dim = features.size(1)
        expected_dim = 8 * 200
        
        # Create projection if needed and not exists
        if actual_dim != expected_dim:
            if self.shap_projection is None:
                self.shap_projection = nn.Linear(actual_dim, expected_dim).to(features.device)
            features = self.shap_projection(features)
            
        
        # Directly pass to SHAP classifier
        return self.shap_classifier(features)
# ======================= DOMAIN ADVERSARIAL LOSS =======================
class DomainAdversarialLoss(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),  # REDUCED SIZE
            nn.ReLU(),  # Changed from ReLU to ReLU (no change needed)
            nn.Linear(32, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, features, domain_labels):
        domain_pred = self.domain_classifier(features)
        return self.loss_fn(domain_pred.squeeze(), domain_labels.float())

# ======================= MAIN TRAINING FUNCTION =======================
def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)
    print_environ()
    print(s)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize critical parameters if not set
    if not hasattr(args, 'latent_domain_num') or args.latent_domain_num is None:
        args.latent_domain_num = 5  # Default value
        print(f"Using default latent_domain_num: {args.latent_domain_num}")
    
    if not hasattr(args, 'bottleneck') or args.bottleneck is None:
        args.bottleneck = 256  # Default value
        print(f"Using default bottleneck dimension: {args.bottleneck}")
    
    # Handle curriculum parameters
    if getattr(args, 'curriculum', False):
        if hasattr(args, 'CL_PHASE_EPOCHS'):
            if isinstance(args.CL_PHASE_EPOCHS, int):
                args.CL_PHASE_EPOCHS = [args.CL_PHASE_EPOCHS] * 3
            elif not isinstance(args.CL_PHASE_EPOCHS, list):
                args.CL_PHASE_EPOCHS = [3, 3, 3]
        else:
            args.CL_PHASE_EPOCHS = [3, 3, 3]
        
        if not hasattr(args, 'CL_DIFFICULTY') or not isinstance(args.CL_DIFFICULTY, list):
            args.CL_DIFFICULTY = [0.2, 0.5, 0.8]
        
        print(f"Curriculum settings: Phases={len(args.CL_PHASE_EPOCHS)}, Epochs per phase: {args.CL_PHASE_EPOCHS}, Difficulties: {args.CL_DIFFICULTY}")
    
    loader_data = get_act_dataloader(args)
    tr = loader_data[4]
    val = loader_data[5]
    targetdata = loader_data[6]
    
    if args.use_gnn and GNN_AVAILABLE:
        LoaderClass = PyGDataLoader
        print("Using PyGDataLoader for GNN data")
    else:
        LoaderClass = TorchDataLoader
        print("Using TorchDataLoader for CNN data")
    
    # Create temporary loader for automated K estimation
    temp_train_loader = LoaderClass(
        dataset=tr,
        batch_size=min(32, len(tr)),  # Use smaller batch size for estimation
        shuffle=True,
        num_workers=min(2, args.N_WORKERS))
    
    # Run automated K estimation if enabled
    if getattr(args, 'automated_k', False):
        print("\nRunning automated K estimation...")
        
        if args.use_gnn and GNN_AVAILABLE:
            print("Using GNN for feature extraction")
            graph_builder = GraphBuilder(
                method='correlation',
                threshold_type='adaptive',
                default_threshold=0.3,
                adaptive_factor=1.5
            )
            temp_model = EnhancedTemporalGCN(
                input_dim=8,
                hidden_dim=args.gnn_hidden_dim,
                output_dim=args.gnn_output_dim,
                graph_builder=graph_builder,
                n_layers=getattr(args, 'gnn_layers', 1),
                use_tcn=getattr(args, 'use_tcn', True)
            ).to(args.device)
        else:
            print("Using CNN for feature extraction")
            temp_model = ActNetwork(args.dataset).to(args.device)
        
        temp_model.eval()
        feature_list = []
        
        with torch.no_grad():
            for batch in temp_train_loader:
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    x = inputs
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    x = inputs
                
                if args.use_gnn and GNN_AVAILABLE:
                    x = transform_for_gnn(x)
                
                features = temp_model(x)
                feature_list.append(features.detach().cpu().numpy())
        
        all_features = np.concatenate(feature_list, axis=0)
        optimal_k = automated_k_estimation(all_features)
        args.latent_domain_num = optimal_k
        print(f"Using automated latent_domain_num (K): {args.latent_domain_num}")
        
        del temp_model
        torch.cuda.empty_cache()
    
    # Adjust batch size based on latent domains
    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num
    print(f"Adjusted batch size: {args.batch_size}")
    
    # Create main data loaders with adjusted batch size
    train_loader = LoaderClass(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=min(2, args.N_WORKERS),
        drop_last=False,
        shuffle=True
    )
    
    train_loader_noshuffle = LoaderClass(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=min(2, args.N_WORKERS),
        drop_last=False,
        shuffle=False
    )
    
    valid_loader = LoaderClass(
        dataset=val,
        batch_size=args.batch_size,
        num_workers=min(2, args.N_WORKERS),
        drop_last=False,
        shuffle=False
    )
    
    target_loader = LoaderClass(
        dataset=targetdata,
        batch_size=args.batch_size,
        num_workers=min(2, args.N_WORKERS),
        drop_last=False,
        shuffle=False
    )
    
    # Create algorithm instance AFTER setting latent_domain_num
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(args.device)
    
    # Configure GNN components if needed
    if args.use_gnn and GNN_AVAILABLE:
        print("\n===== Initializing GNN Feature Extractor =====")
        
        graph_builder = GraphBuilder(
            method='correlation',
            threshold_type='adaptive',
            default_threshold=0.3,
            adaptive_factor=1.5,
            fully_connected_fallback=True
        )
        
        args.gnn_layers = getattr(args, 'gnn_layers', 1)
        args.use_tcn = getattr(args, 'use_tcn', True)
        
        gnn_model = EnhancedTemporalGCN(
            input_dim=8,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            graph_builder=graph_builder,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            bidirectional=args.bidirectional,
            n_layers=args.gnn_layers,
            use_tcn=args.use_tcn
        ).to(args.device)
        
        algorithm.featurizer = gnn_model
        
        # Set consistent bottleneck dimensions
        args.bottleneck = 256
        args.bottleneck_dim = 256
        
        # Create consistent bottlenecks
        algorithm.bottleneck = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Flatten()
        ).cuda()
        
        algorithm.abottleneck = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Flatten()
        ).cuda()
        
        algorithm.dbottleneck = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Flatten()
        ).cuda()
        
        print(f"Bottleneck: 256 -> 256")
        
        # GNN pretraining
        if hasattr(args, 'gnn_pretrain_epochs') and args.gnn_pretrain_epochs > 0:
            print(f"\n==== GNN Pretraining ({args.gnn_pretrain_epochs} epochs) ====")
            gnn_optimizer = torch.optim.Adam(
                algorithm.featurizer.parameters(),
                lr=args.gnn_lr,
                weight_decay=0.0
            )
            
            for epoch in range(args.gnn_pretrain_epochs):
                gnn_model.train()
                total_loss = 0.0
                batch_count = 0
                for batch in train_loader:
                    if args.use_gnn and GNN_AVAILABLE:
                        inputs = batch[0].to(args.device)
                        labels = batch[1].to(args.device)
                        domains = batch[2].to(args.device)
                        x = inputs
                    else:
                        inputs = batch[0].to(args.device).float()
                        labels = batch[1].to(args.device).long()
                        domains = batch[2].to(args.device).long()
                        x = inputs
                    
                    if args.use_gnn and GNN_AVAILABLE:
                        x = transform_for_gnn(x)
                    
                    # Compute target: mean along time dimension
                    if x.dim() == 3:  # [batch, time, features]
                        target = torch.mean(x, dim=1)
                    elif x.dim() == 4:  # [batch, channels, 1, time]
                        x_processed = x.squeeze(2).permute(0, 2, 1)
                        target = torch.mean(x_processed, dim=1)
                    else:
                        target = torch.mean(x, dim=1)

                    features = gnn_model(x)

                    # Add reconstruction head if needed
                    if not hasattr(gnn_model, 'reconstruction_head'):
                        target_dim = target.shape[1]
                        gnn_model.reconstruction_head = nn.Sequential(
                            nn.Linear(args.gnn_output_dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, target_dim)
                        ).to(args.device)
                        print(f"Created reconstruction head with output dim: {target_dim}")

                    reconstructed = gnn_model.reconstruction_head(features)
                    loss = torch.nn.functional.mse_loss(reconstructed, target)
                    
                    if torch.isnan(loss):
                        continue
                    
                    gnn_optimizer.zero_grad()
                    loss.backward()
                    gnn_optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    avg_loss = total_loss / batch_count
                    print(f'GNN Pretrain Epoch {epoch+1}/{args.gnn_pretrain_epochs}: Loss {avg_loss:.4f}')
            
            print("GNN pretraining complete")
    algorithm.create_teacher = types.MethodType(lambda self: None, algorithm)
    algorithm.update_teacher = types.MethodType(
        Diversify.update_teacher, 
        algorithm
    )
    algorithm.update_teacher()
    
    algorithm.train()
    
    # Configure optimizers
    optimizer = algorithm.configure_optimizers(args)
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')
    
    augmenter = EMGDataAugmentation(args).cuda()
    
    if getattr(args, 'domain_adv_weight', 0.0) > 0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(
            bottleneck_dim=int(args.bottleneck)
        ).cuda()
        print(f"Added domain adversarial training (weight: {args.domain_adv_weight})")
    
    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'ent_loss',
                            'total_loss', 'train_acc', 'valid_acc', 'target_acc',
                            'total_cost_time', 'h_divergence', 'domain_acc', 'contrast_loss']}
    best_valid_acc, target_acc = 0, 0
    
    entire_source_loader = LoaderClass(
        tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.N_WORKERS))
    
    best_valid_acc = 0
    epochs_without_improvement = 0
    early_stopping_patience = getattr(args, 'early_stopping_patience', 20)
    
    MAX_GRAD_NORM = 1.0
    
    def evaluate_accuracy(loader):
        correct = 0
        total = 0
        algorithm.eval()
        with torch.no_grad():
            for batch in loader:
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    inputs = transform_for_gnn(inputs)
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                
                outputs = algorithm.predict(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    
    global_step = 0
    for round_idx in range(args.max_epoch):
        if hasattr(algorithm.featurizer, 'dropout'):
            algorithm.featurizer.dropout.p = 0.0
        
        print(f'\n======== ROUND {round_idx} ========')
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {round_idx}")
            break
            
        if getattr(args, 'curriculum', False) and round_idx < len(args.CL_PHASE_EPOCHS):
            current_epochs = args.CL_PHASE_EPOCHS[round_idx]
            current_difficulty = args.CL_DIFFICULTY[round_idx]
            print(f"\nCurriculum Stage {round_idx+1}/{len(args.CL_PHASE_EPOCHS)}")
            print(f"Difficulty: {current_difficulty:.1f}, Epochs: {current_epochs}")
            
            algorithm.eval()
            
            transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
    
            class CurriculumEvaluator:
                def __init__(self, algorithm, transform_fn=None):
                    self.algorithm = algorithm
                    self.transform_fn = transform_fn
                    
                def eval(self):
                    self.algorithm.eval()
                    
                def predict(self, x):
                    if self.transform_fn:
                        x = self.transform_fn(x)
                    return self.algorithm.predict(x)
                    
                @property
                def featurizer(self):
                    return self.algorithm.featurizer
            
            evaluator = CurriculumEvaluator(algorithm, transform_fn)
            
            curriculum_dataset = get_curriculum_loader(
                args,
                evaluator,
                tr,
                val,
                stage=round_idx
            )
            train_loader = LoaderClass(
                curriculum_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=min(2, args.N_WORKERS),
                drop_last=False
            )
            train_loader_noshuffle = LoaderClass(
                curriculum_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=min(1, args.N_WORKERS),
                drop_last=False
            )
            algorithm.train()
        else:
            current_epochs = args.local_epoch
            print(f"\nStandard Training Stage")
            print(f"Epochs: {current_epochs}")
        
        print('\n==== Feature update ====')
        print_row(['epoch', 'class_loss'], colwidth=15)
        for step in range(current_epochs):
            epoch_class_loss = 0.0
            batch_count = 0
            algorithm.train()
            for batch in train_loader:
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    data = [inputs, labels, domains]
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    data = [inputs, labels, domains]
                
                loss_result_dict = algorithm.update_a(data, opta)
                
                if not np.isfinite(loss_result_dict['class']):
                    continue
                
                epoch_class_loss += loss_result_dict['class']
                batch_count += 1
                
            if batch_count > 0:
                epoch_class_loss /= batch_count
                print_row([step, epoch_class_loss], colwidth=15)
                logs['class_loss'].append(epoch_class_loss)
            
        print('\n==== Latent domain characterization ====')
        print_row(['epoch', 'total_loss', 'dis_loss', 'ent_loss'], colwidth=15)
        for step in range(current_epochs):
            epoch_total = 0.0
            epoch_dis = 0.0
            epoch_ent = 0.0
            batch_count = 0
            
            algorithm.train()
            for batch in train_loader:
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    data = [inputs, labels, domains]
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    data = [inputs, labels, domains]
                
                loss_result_dict = algorithm.update_d(data, optd)
                
                if any(not np.isfinite(v) for v in loss_result_dict.values()):
                    continue
                    
                epoch_total += loss_result_dict['total']
                epoch_dis += loss_result_dict['dis']
                epoch_ent += loss_result_dict['ent']
                batch_count += 1
                
            if batch_count > 0:
                epoch_total /= batch_count
                epoch_dis /= batch_count
                epoch_ent /= batch_count
                print_row([step, epoch_total, epoch_dis, epoch_ent], colwidth=15)
                logs['dis_loss'].append(epoch_dis)
                logs['ent_loss'].append(epoch_ent)
                logs['total_loss'].append(epoch_total)
                
        algorithm.set_dlabel(train_loader)
        
        print('\n==== Domain-invariant feature learning ====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch', 'class_loss', 'dis_loss', 'contrast_loss', 
                     'train_acc', 'valid_acc', 'target_acc', 'total_cost_time']
        print_row(print_key, colwidth=15)
        
        round_start_time = time.time()
        for step in range(current_epochs):
            step_start_time = time.time()
            
            algorithm.train()
            for batch in train_loader:
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    data = [inputs, labels, domains]
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    data = [inputs, labels, domains]

                step_vals = algorithm.update(data, optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.parameters(), MAX_GRAD_NORM)
            
            algorithm.eval()
            train_acc = evaluate_accuracy(train_loader_noshuffle)
            valid_acc = evaluate_accuracy(valid_loader)
            target_acc = evaluate_accuracy(target_loader)
            
            results = {
                'epoch': global_step,
                'train_acc': train_acc,
                'valid_acc': valid_acc,
                'target_acc': target_acc,
                'total_cost_time': time.time() - step_start_time
            }
            
            if 'contrast' in step_vals:
                results['contrast_loss'] = step_vals['contrast']
                logs['contrast_loss'].append(step_vals['contrast'])
            
            for key in ['class', 'dis']:
                if key in step_vals:
                    results[f"{key}_loss"] = step_vals[key]
                    logs[f"{key}_loss"].append(step_vals[key])
            
            for metric in ['train_acc', 'valid_acc', 'target_acc']:
                logs[metric].append(results[metric])
            
            algorithm.update_teacher()
            
            if algorithm.scheduler is not None:
                algorithm.scheduler.step(valid_acc)
            
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
                epochs_without_improvement = 0
                torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))
            else:
                epochs_without_improvement += 1
                
            row = [
                results.get('epoch', global_step),
                results.get('class_loss', 0),
                results.get('dis_loss', 0),
                results.get('contrast_loss', 0),
                results.get('train_acc', 0),
                results.get('valid_acc', 0),
                results.get('target_acc', 0),
                results.get('total_cost_time', 0)
            ]
            print_row(row, colwidth=15)
            global_step += 1
            
        logs['total_cost_time'].append(time.time() - round_start_time)
        
        if round_idx % 5 == 0:
            print("\nCalculating h-divergence...")
            algorithm.eval()
            
            source_features = []
            with torch.no_grad():
                for data in entire_source_loader:
                    if args.use_gnn and GNN_AVAILABLE:
                        inputs = data[0].to(args.device)
                        inputs = transform_for_gnn(inputs)
                    else:
                        inputs = data[0].to(args.device).float()
                    
                    features = algorithm.featurizer(inputs).detach().cpu().numpy()
                    source_features.append(features)
            source_features = np.concatenate(source_features, axis=0)
            
            target_features = []
            with torch.no_grad():
                for data in target_loader:
                    if args.use_gnn and GNN_AVAILABLE:
                        inputs = data[0].to(args.device)
                        inputs = transform_for_gnn(inputs)
                    else:
                        inputs = data[0].to(args.device).float()
                    
                    features = algorithm.featurizer(inputs).detach().cpu().numpy()
                    target_features.append(features)
            target_features = np.concatenate(target_features, axis=0)
            
            h_div, domain_acc = calculate_h_divergence(source_features, target_features)
            logs['h_divergence'].append(h_div)
            logs['domain_acc'].append(domain_acc)
            print(f" H-Divergence: {h_div:.4f}, Domain Classifier Acc: {domain_acc:.4f}")
            
            algorithm.train()
            
    print(f'\n🎯 Final Target Accuracy: {target_acc:.4f}')
    
    if getattr(args, 'enable_shap', False):
        print("\n📊 Running SHAP explainability...")
        try:
            # Prepare background and evaluation data
            background_list = []
            if args.use_gnn and GNN_AVAILABLE:
                print("Collecting SHAP background data for GNN...")
                print(f"Loader type: {type(valid_loader)}")
                
                for batch_idx, batch in enumerate(valid_loader):
                    # Handle PyG Batch objects
                    if isinstance(batch, (Data, Batch)) or hasattr(batch, 'to_data_list'):
                        try:
                            if hasattr(batch, 'to_data_list'):
                                data_list = batch.to_data_list()
                            else:
                                data_list = [batch]
                            background_list.extend(data_list)
                        except Exception as e:
                            print(f"Error converting batch: {str(e)}")
                    
                    # Handle tuple format
                    elif isinstance(batch, (tuple, list)):
                        for item in batch:
                            if isinstance(item, (Data, Batch)) or hasattr(item, 'to_data_list'):
                                try:
                                    if hasattr(item, 'to_data_list'):
                                        data_list = item.to_data_list()
                                    else:
                                        data_list = [item]
                                    background_list.extend(data_list)
                                except Exception as e:
                                    print(f"Error converting DataBatch: {str(e)}")
                                break
                    
                    if len(background_list) >= 64:
                        break
                
                if background_list:
                    # For GNN: use first sample as background
                    background = background_list[0]
                    X_eval = Batch.from_data_list(background_list[:10])
                    # Add debug prints
                    print(f"Background sample node features shape: {background.x.shape}")
                    print(f"Evaluation batch node features shape: {X_eval.x.shape}")
                    
                    print(f"Using first sample as background for GNN")
                    print(f"Created evaluation batch with {len(background_list[:10])} graphs")
                else:
                    print("⚠️ Couldn't collect background data for SHAP")
                    background = None
                    X_eval = None
            else:
                # Standard tensor handling for CNN
                background = get_background_batch(valid_loader, size=64).to(args.device)
                X_eval = background[:10]
                print(f"Collected CNN background shape: {background.shape}")
            
            if background is None or X_eval is None:
                print("Skipping SHAP due to missing data")
            else:
                # Disable inplace operations in the model
                disable_inplace_relu(algorithm)
                
                # Create a prediction wrapper
                class UnifiedPredictor(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        
                    def forward(self, x):
                        if isinstance(x, (Data, Batch)) or hasattr(x, 'to_data_list'):
                            return self.model.predict(x)
                        elif isinstance(x, list) and isinstance(x[0], Data):
                            batch = Batch.from_data_list(x)
                            return self.model.predict(batch)
                        else:
                            return self.model.predict(x)
                    
                    # ADD THIS METHOD TO FIX PREDICTION ISSUE
                    def predict(self, x):
                        return self.forward(x)
    
                # Set SHAP mode in the GNN model
                if args.use_gnn and GNN_AVAILABLE:
                    algorithm.featurizer.shap_mode = True
                # Create the unified predictor
                unified_predictor = UnifiedPredictor(algorithm).to(args.device)
                unified_predictor.eval()
                
                # Move data to device
                background = background.to(args.device)
                X_eval = X_eval.to(args.device)
                
                # Compute SHAP values safely
                print("Computing SHAP values...")
                shap_explanation = safe_compute_shap_values(unified_predictor, background, X_eval)
                # Revert SHAP mode
                if args.use_gnn and GNN_AVAILABLE:
                    algorithm.featurizer.shap_mode = False
                # Check if SHAP computation succeeded
                if shap_explanation is None:
                    print("⚠️ SHAP computation failed. Skipping analysis.")
                else:
                    # Extract values from Explanation object
                    shap_vals = shap_explanation.values
                    print(f"SHAP values shape: {shap_vals.shape}")

                    # ==== ADD THIS AGGREGATION BLOCK HERE ====
                    # Create aggregated version for visualizations
                    if shap_vals.ndim == 3:  # (samples, timesteps, classes)
                        shap_vals_agg = np.abs(shap_vals).max(axis=-1)
                        print(f"Aggregated SHAP values shape: {shap_vals_agg.shape}")
                    else:
                        shap_vals_agg = shap_vals.copy()

                    if args.use_gnn and GNN_AVAILABLE:
                        # For GNN, convert batched graph to numpy
                        X_eval_np = X_eval.x.detach().cpu().numpy()
                    else:
                        # For standard models
                        X_eval_np = X_eval.detach().cpu().numpy()
                    # Debug print sample data
                    print(f"Sample SHAP values (min, max, mean): {shap_vals_agg.min()}, {shap_vals_agg.max()}, {shap_vals_agg.mean()}")
                    print(f"Sample signal data (min, max, mean): {X_eval_np.min()}, {X_eval_np.max()}, {X_eval_np.mean()}")
                    # ==== END OF AGGREGATION BLOCK ====
                    
                    
                    # Handle GNN dimensionality for visualization
                    if args.use_gnn and GNN_AVAILABLE:
                        print(f"Original SHAP values shape: {shap_vals.shape}")
                        print(f"Original X_eval shape: {X_eval_np.shape}")
                        
                        # If 4D, reduce to 3D by summing over classes
                        if shap_vals.ndim == 4:
                            # Sum across classes to get overall feature importance
                            shap_vals = np.abs(shap_vals).sum(axis=-1)
                            print(f"SHAP values after class sum: {shap_vals.shape}")
                        
                        # Now we should have 3D: [batch, time, channels]
                        if shap_vals.ndim == 3:
                            # Convert to [batch, channels, 1, time] for visualization
                            shap_vals = np.transpose(shap_vals, (0, 2, 1))
                            shap_vals = np.expand_dims(shap_vals, axis=2)
                            
                            X_eval_np = np.transpose(X_eval_np, (0, 2, 1))
                            X_eval_np = np.expand_dims(X_eval_np, axis=2)
                        else:
                            print(f"⚠️ Unexpected SHAP values dimension: {shap_vals.ndim}")
                        
                        # Visualize the first sample
                        try:
                            plot_emg_shap_4d(X_eval_np[0], shap_vals[0], 
                                             output_path=os.path.join(args.output, "shap_gnn_sample.html"))
                        except Exception as e:
                            print(f"4D plot failed: {str(e)}")
                    
                    # Generate core visualizations for ALL models (including GNN)
                    try:
                        # USE shap_vals_agg HERE
                        plot_summary(shap_vals_agg, X_eval_np, 
                                    output_path=os.path.join(args.output, "shap_summary.png"))
                    except Exception as e:
                        print(f"Summary plot failed: {str(e)}")
                    
                    try:
                        # USE shap_vals_agg HERE
                        overlay_signal_with_shap(X_eval_np[0], shap_vals_agg, 
                                                output_path=os.path.join(args.output, "shap_overlay.png"))
                    except Exception as e:
                        print(f"Signal overlay failed: {str(e)}")
                    
                    try:
                        # USE shap_vals_agg HERE
                        plot_shap_heatmap(shap_vals_agg, 
                                         output_path=os.path.join(args.output, "shap_heatmap.png"))
                    except Exception as e:
                        print(f"Heatmap failed: {str(e)}")
                    
                    # Evaluate SHAP impact
                    try:
                        print("Evaluating SHAP impact...")
                        base_preds, masked_preds, acc_drop = evaluate_shap_impact(unified_predictor, X_eval, shap_vals)
                        
                        if base_preds is not None and masked_preds is not None:
                            # Save SHAP values
                            save_path = os.path.join(args.output, "shap_values.npy")
                            save_shap_numpy(shap_vals, save_path=save_path)
                            
                            # Compute impact metrics
                            print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
                            print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
                            print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
                            try:
                                print(f"[SHAP] AOPC: {compute_aopc(unified_predictor, X_eval, shap_vals):.4f}")
                            except:
                                print("AOPC computation failed")
                    except Exception as e:
                        print(f"SHAP impact evaluation failed: {str(e)}")
                    
                    # Compute advanced metrics
                    try:
                        metrics = evaluate_advanced_shap_metrics(shap_vals, X_eval)
                        print(f"[SHAP] Entropy: {metrics.get('shap_entropy', 0):.4f}")
                        print(f"[SHAP] Coherence: {metrics.get('feature_coherence', 0):.4f}")
                        print(f"[SHAP] Channel Variance: {metrics.get('channel_variance', 0):.4f}")
                        print(f"[SHAP] Temporal Entropy: {metrics.get('temporal_entropy', 0):.4f}")
                        print(f"[SHAP] Mutual Info: {metrics.get('mutual_info', 0):.4f}")
                        print(f"[SHAP] PCA Alignment: {metrics.get('pca_alignment', 0):.4f}")
                    except Exception as e:
                        print(f"Advanced metrics failed: {str(e)}")
                    
                    # Compute similarity metrics between first two samples
                    try:
                        shap_array = _get_shap_array(shap_vals)
                        if len(shap_array) >= 2:
                            # Extract SHAP values for first two samples
                            sample1 = shap_array[0]
                            sample2 = shap_array[1]
                            
                            print(f"[SHAP] Jaccard (top-10): {compute_jaccard_topk(sample1, sample2, k=10):.4f}")
                            print(f"[SHAP] Kendall's Tau: {compute_kendall_tau(sample1, sample2):.4f}")
                            print(f"[SHAP] Cosine Similarity: {cosine_similarity_shap(sample1, sample2):.4f}")
                        else:
                            print("[SHAP] Not enough samples for similarity metrics")
                    except Exception as e:
                        print(f"Similarity metrics failed: {str(e)}")
                    
                    # Generate 4D visualizations for non-GNN models
                    if not (args.use_gnn and GNN_AVAILABLE):
                        try:
                            # For scatter plot - use first sample
                            plot_emg_shap_4d(
                                X_eval_np[0], 
                                shap_vals[0] if shap_vals.ndim > 2 else shap_vals_agg[0],
                                output_path=os.path.join(args.output, "shap_4d_scatter.html")
                            )
                        except Exception as e:
                            print(f"4D scatter plot failed: {str(e)}")
                        
                        try:
                            # For surface plot
                            plot_4d_shap_surface(
                                shap_vals if shap_vals.ndim > 2 else shap_vals_agg,
                                output_path=os.path.join(args.output, "shap_4d_surface.html")
                            )
                        except Exception as e:
                            print(f"4D surface plot failed: {str(e)}")
                    
                    # Confusion matrix
                    try:
                        print("Generating confusion matrix...")
                        true_labels, pred_labels = [], []
                        device = next(unified_predictor.parameters()).device  # Get model device
    
                        for data in valid_loader:
                            if args.use_gnn and GNN_AVAILABLE:
                                # Handle different data formats
                                if isinstance(data, Data):
                                    x = data
                                    y = data.y
                                elif hasattr(data, 'y'):
                                    # Batch object
                                    x = data
                                    y = data.y
                                elif isinstance(data, (list, tuple)) and len(data) >= 2:
                                    # Standard format: [inputs, labels, ...]
                                    x = data[0]
                                    y = data[1]
                                else:
                                    # Unsupported format
                                    continue
                            else:
                                x = data[0].float()  # Remove .to(device) here
                                y = data[1]
                            
                            # Move data to model's device
                            if isinstance(x, (Data, Batch)):
                                x = x.to(device)
                            elif isinstance(x, torch.Tensor):
                                x = x.to(device)
                            
                            with torch.no_grad():
                                # Use UnifiedPredictor's predict method
                                preds = unified_predictor.predict(x).cpu()
                                
                                # Handle single values vs tensors
                                if isinstance(y, torch.Tensor):
                                    y = y.cpu().numpy()
                                elif isinstance(y, list):
                                    y = np.array(y)
                                else:
                                    y = np.array([y])
                                    
                                true_labels.extend(y)
                                pred_labels.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())
                        
                        cm = confusion_matrix(true_labels, pred_labels)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                        disp.plot(cmap="Blues")
                        plt.title("Confusion Matrix (Validation Set)")
                        plt.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=300)
                        plt.close()
                        
                        print("✅ SHAP analysis completed successfully")
                    except Exception as e:
                        print(f"Confusion matrix failed: {str(e)}")
        except Exception as e:
            print(f"[ERROR] SHAP analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
        # ======================= END SHAP SECTION =======================
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot losses if available
        if logs['class_loss']:
            plt.subplot(2, 1, 1)
            loss_epochs = list(range(len(logs['class_loss'])))
            plt.plot(loss_epochs, logs['class_loss'], label="Class Loss", marker='o')
            
            if len(logs['dis_loss']) == len(loss_epochs):
                plt.plot(loss_epochs, logs['dis_loss'], label="Dis Loss", marker='x')
            
            if len(logs['total_loss']) == len(loss_epochs):
                plt.plot(loss_epochs, logs['total_loss'], label="Total Loss", linestyle='--')
            
            plt.title("Losses over Training Steps")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
        
        # Plot accuracies if available
        if logs['train_acc']:
            plt.subplot(2, 1, 2)
            acc_epochs = list(range(len(logs['train_acc'])))
            plt.plot(acc_epochs, logs['train_acc'], label="Train Accuracy", marker='o')
            
            if len(logs['valid_acc']) == len(acc_epochs):
                plt.plot(acc_epochs, logs['valid_acc'], label="Valid Accuracy", marker='x')
            
            if len(logs['target_acc']) == len(acc_epochs):
                plt.plot(acc_epochs, logs['target_acc'], label="Target Accuracy", linestyle='--')
            
            plt.title("Accuracy over Training Steps")
            plt.xlabel("Global Step")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "training_metrics.png"), dpi=300)
        plt.close()
        print("✅ Training metrics plot saved")
        
        # Domain discrepancy plot remains unchanged
        if logs['h_divergence']:
            plt.figure(figsize=(10, 6))
            h_epochs = [i * 5 for i in range(len(logs['h_divergence']))]
            plt.plot(h_epochs, logs['h_divergence'], 'o-', label='H-Divergence')
            
            if len(logs['domain_acc']) == len(h_epochs):
                plt.plot(h_epochs, logs['domain_acc'], 's-', label='Domain Classifier Acc')
            
            plt.title("Domain Discrepancy over Training")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(args.output, "domain_discrepancy.png"), dpi=300)
            plt.close()
            print("✅ Domain discrepancy plot saved")
            
    except Exception as e:
        print(f"[WARNING] Failed to generate training plots: {str(e)}")

if __name__ == '__main__':
    args = get_args()
    # ====================== OPTIMIZED PARAMETER SETTINGS ======================
    # All regularization removed
    args.lambda_cls = getattr(args, 'lambda_cls', 1.0)
    args.lambda_dis = getattr(args, 'lambda_dis', 0.0001)  # MINIMAL
    args.label_smoothing = 0.0  # DISABLED
    args.max_grad_norm = 1.0  # Loosened gradient clipping
    args.gnn_pretrain_epochs = getattr(args, 'gnn_pretrain_epochs', 5)  # Enabled with 5 epochs
    
    # CRITICAL FIX: Add local_epoch parameter
    args.local_epoch = getattr(args, 'local_epoch', 5)  # Default to 5 epochs per phase

    # GNN parameters minimized
    if not hasattr(args, 'use_gnn'):
        args.use_gnn = False
        
    if args.use_gnn:
        if not GNN_AVAILABLE:
            print("[WARNING] GNN requested but not available. Falling back to CNN.")
            args.use_gnn = False
        else:
            args.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 128)  # Increased
            args.gnn_output_dim = getattr(args, 'gnn_output_dim', 256)  # Increased
            args.gnn_layers = 1  # MINIMAL LAYERS
            args.gnn_lr = getattr(args, 'gnn_lr', 0.01)  # Increased
            args.gnn_weight_decay = 0.0  # DISABLED
            
            args.use_tcn = getattr(args, 'use_tcn', True)
            args.lstm_hidden_size = 128  # Increased
            args.lstm_layers = 1
            args.bidirectional = True  # ENABLED
            args.lstm_dropout = 0.0  # DISABLED

    # Optimizer settings minimized
    args.optimizer = getattr(args, 'optimizer', 'adam')
    args.weight_decay = 1e-3  # Enable weight decay
    args.domain_adv_weight = 0.1  # Enable domain adaptation
    args.lr = 0.01  # Increased learning rate

    # Augmentation enabled for contrastive learning
    args.jitter_scale = 0.5
    args.scaling_std = 0.5
    args.warp_ratio = 0.5
    args.aug_prob = 0.9

    # Training schedule minimized
    args.max_epoch = getattr(args, 'max_epoch', 100)  # INCREASED
    args.early_stopping_patience = 30  # INCREASED

    # Domain adaptation minimized
    if not hasattr(args, 'adv_weight'):
        args.adv_weight = 0.1  # DISABLED

    main(args)
