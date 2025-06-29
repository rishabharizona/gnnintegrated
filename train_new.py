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
    # Filter out NaN values
    mask_source = ~np.isnan(features_source).any(axis=1)
    mask_target = ~np.isnan(features_target).any(axis=1)
    
    features_source = features_source[mask_source]
    features_target = features_target[mask_target]
    
    if len(features_source) == 0 or len(features_target) == 0:
        print("Warning: No valid features for h-divergence calculation")
        return 0.0, 0.5
    
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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.bn = nn.BatchNorm1d(n_outputs)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
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
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        return self.activation(out + residual)

# ======================= DATA AUGMENTATION MODULE =======================
class EMGDataAugmentation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.jitter_scale = 0.1
        self.scaling_std = 0.1
        self.warp_ratio = 0.1
        self.aug_prob = 0.5

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
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
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
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation(x)
        x = self.dropout(x)
        return x

# ======================= ENHANCED GNN ARCHITECTURE =======================
class EnhancedTemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder, 
                 n_layers=1, use_tcn=True, lstm_hidden_size=64, 
                 lstm_layers=1, bidirectional=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_builder = graph_builder
        self.n_layers = n_layers
        self.use_tcn = use_tcn
        
        # Feature projection if needed
        self.feature_projection = None
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            layer = TemporalGCNLayer(
                input_dim=input_dim if i == 0 else hidden_dim,
                output_dim=hidden_dim,
                graph_builder=graph_builder
            )
            self.gnn_layers.append(layer)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Temporal processing
        if use_tcn:
            tcn_layers = []
            num_channels = [hidden_dim] * 2
            kernel_size = 3
            for i in range(len(num_channels)):
                dilation = 2 ** i
                in_channels = hidden_dim if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
                tcn_layers += [TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation, dropout=0.1
                )]
            self.tcn = nn.Sequential(*tcn_layers)
            self.tcn_proj = nn.Linear(num_channels[-1], output_dim)
        else:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.1
            )
            lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
            self.lstm_proj = nn.Linear(lstm_output_dim, output_dim)
        
        # Skip connection
        self.skip_conn = nn.Linear(hidden_dim, output_dim)
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.gnn_layers:
            if hasattr(layer.linear, 'weight'):
                nn.init.xavier_uniform_(layer.linear.weight)
        for name, param in self.named_parameters():
            if 'weight' in name and 'bn' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
    def forward(self, x):
        # Handle input types and dimensions
        if isinstance(x, (Data, Batch)):
            from torch_geometric.utils import to_dense_batch
            x, mask = to_dense_batch(x.x, x.batch)
        
        if x.dim() == 4:
            # [batch, channels, 1, time] -> [batch, time, channels]
            x = x.squeeze(2).permute(0, 2, 1)
        elif x.dim() == 3:
            if x.size(1) == 8 or x.size(1) == 200:
                # [batch, channels, time] -> [batch, time, channels]
                x = x.permute(0, 2, 1)
            elif x.size(2) == 8 or x.size(2) == 200:
                # Already [batch, time, channels]
                pass
        
        # Feature projection if needed
        if x.size(-1) == 200 and self.input_dim == 8:
            if self.feature_projection is None:
                self.feature_projection = nn.Linear(200, 8).to(x.device)
            x = self.feature_projection(x)
        
        # GNN processing
        for layer in self.gnn_layers:
            x = layer(x)
            x = F.relu(x)
        
        # Attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        
        # Temporal processing
        if self.use_tcn:
            # [batch, time, channels] -> [batch, channels, time]
            tcn_in = x.permute(0, 2, 1)
            tcn_out = self.tcn(tcn_in)
            # [batch, channels, time] -> [batch, time, channels]
            tcn_out = tcn_out.permute(0, 2, 1)
            temporal_out = self.tcn_proj(tcn_out)
        else:
            lstm_out, _ = self.lstm(x)
            temporal_out = self.lstm_proj(lstm_out)
        
        # Pooling
        gnn_out = temporal_out.mean(dim=1)
        skip_out = self.skip_conn(x.mean(dim=1))
        
        # Gated fusion
        gate = torch.sigmoid(0.5 * gnn_out + 0.5 * skip_out)
        output = gate * gnn_out + (1 - gate) * skip_out
        
        return output

# ======================= DATA NORMALIZATION =======================
def compute_dataset_mean_std(dataloader, device):
    """Compute mean and std for dataset normalization"""
    mean = torch.zeros(1).to(device)
    std = torch.zeros(1).to(device)
    total_samples = 0
    
    for batch in dataloader:
        if args.use_gnn and GNN_AVAILABLE:
            inputs = batch[0].to(device)
        else:
            inputs = batch[0].to(device).float()
            
        # Flatten all dimensions except batch
        b = inputs.size(0)
        inputs = inputs.view(b, -1)
        
        total_samples += b
        mean += inputs.mean(dim=1).sum()
        std += inputs.std(dim=1).sum()
    
    mean /= total_samples
    std /= total_samples
    
    # Add epsilon to avoid division by zero
    std = torch.clamp(std, min=1e-6)
    
    print(f"Computed normalization: mean={mean.item():.4f}, std={std.item():.4f}")
    return mean, std

# ======================= SAFE LOSS FUNCTIONS =======================
def safe_cross_entropy(logits, targets, eps=1e-6):
    """Numerically stable cross entropy with label smoothing"""
    log_probs = F.log_softmax(logits, dim=1)
    targets = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1)
    targets = (1 - eps) * targets + eps / logits.size(1)
    return -(targets * log_probs).sum(dim=1).mean()

# ======================= MAIN TRAINING FUNCTION =======================
def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)
    print_environ()
    print(s)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    os.makedirs(args.output, exist_ok=True)
    
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
    
    temp_train_loader = LoaderClass(
        dataset=tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(2, args.N_WORKERS))
    
    # CRITICAL FIX: Ensure latent_domain_num is set before algorithm creation
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
                    x = transform_for_gnn(inputs)
                else:
                    inputs = batch[0].to(args.device).float()
                    x = inputs
                
                features = temp_model(x)
                feature_list.append(features.detach().cpu().numpy())
        
        all_features = np.concatenate(feature_list, axis=0)
        optimal_k = automated_k_estimation(all_features)
        args.latent_domain_num = optimal_k
        print(f"Using automated latent_domain_num (K): {args.latent_domain_num}")
        
        del temp_model
        torch.cuda.empty_cache()
    
    # Ensure latent_domain_num is set
    if not hasattr(args, 'latent_domain_num') or args.latent_domain_num is None:
        args.latent_domain_num = 3  # Default value
        print(f"Using default latent_domain_num: {args.latent_domain_num}")
    
    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num
    print(f"Adjusted batch size: {args.batch_size}")
    
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
    
    # ==================== COMPUTE DATASET STATS FOR NORMALIZATION ====================
    print("\nComputing dataset statistics for normalization...")
    train_mean, train_std = compute_dataset_mean_std(train_loader_noshuffle, args.device)
    
    # ==================== MODEL INITIALIZATION ====================
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(args.device)
    
    # Add normalization buffers to model
    algorithm.register_buffer('data_mean', train_mean)
    algorithm.register_buffer('data_std', train_std)
    
    # ==================== ARCHITECTURE ENHANCEMENTS ====================
    # Enhance CNN architecture if not using GNN
    if not args.use_gnn or not GNN_AVAILABLE:
        print("\nEnhancing CNN architecture...")
        # Increase CNN capacity
        algorithm.featurizer.conv1 = nn.Conv2d(
            8, 128, kernel_size=(1, 5), 
            stride=(1, 1), padding=(0, 2)
        algorithm.featurizer.conv2 = nn.Conv2d(
            128, 256, kernel_size=(1, 5), 
            stride=(1, 1), padding=(0, 2))
        
        # Adjust linear layer for new dimensions
        with torch.no_grad():
            dummy_input = torch.randn(2, 8, 1, 200).to(args.device)
            dummy_features = algorithm.featurizer(dummy_input)
            new_feature_dim = dummy_features.shape[1]
            
        algorithm.classifier = nn.Linear(new_feature_dim, args.num_classes).to(args.device)
        print(f"Enhanced CNN: new feature dim={new_feature_dim}")
    
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
        
        # Increase GNN capacity
        args.gnn_hidden_dim = 256
        args.gnn_output_dim = 512
        
        gnn_model = EnhancedTemporalGCN(
            input_dim=8,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            graph_builder=graph_builder,
            n_layers=args.gnn_layers,
            use_tcn=args.use_tcn
        ).to(args.device)
        
        algorithm.featurizer = gnn_model
        
        def create_bottleneck(input_dim, output_dim, layer_spec):
            try:
                num_layers = int(layer_spec)
                layers = []
                current_dim = input_dim
                
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(current_dim, current_dim))
                    layers.append(nn.BatchNorm1d(current_dim))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(0.1))
                
                layers.append(nn.Linear(current_dim, output_dim))
                return nn.Sequential(*layers)
            except ValueError:
                return nn.Sequential(nn.Linear(input_dim, output_dim))
        
        input_dim = args.gnn_output_dim
        output_dim = int(args.bottleneck)
        
        algorithm.bottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        algorithm.abottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        algorithm.dbottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        
        print(f"Created bottlenecks: {input_dim} -> {output_dim}")
        print(f"Bottleneck architecture: {algorithm.bottleneck}")
    
    algorithm.train()
    
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5, verbose=True)
    
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
    early_stopping_patience = getattr(args, 'early_stopping_patience', 10)
    
    MAX_GRAD_NORM = 1.0
    
    # ======================= FIXED EVALUATION FUNCTION =======================
    def evaluate_accuracy(loader):
        correct = 0
        total = 0
        algorithm.eval()
        with torch.no_grad():
            for batch in loader:
                # Handle different data formats consistently
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    # Apply normalization
                    inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    # Always apply transformation for GNN
                    inputs = transform_for_gnn(inputs)
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    # Apply normalization
                    inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                
                # Ensure correct dimensions
                if hasattr(algorithm, 'ensure_correct_dimensions'):
                    inputs = algorithm.ensure_correct_dimensions(inputs)
                
                outputs = algorithm.predict(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    # ======================= END FIXED EVALUATION FUNCTION =======================
    
    global_step = 0
    for round_idx in range(args.max_epoch):
        if hasattr(algorithm.featurizer, 'dropout'):
            algorithm.featurizer.dropout.p = 0.1
        
        print(f'\n======== ROUND {round_idx} ========')
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {round_idx}")
            break
            
        # Set current_epochs based on curriculum or default
        if getattr(args, 'curriculum', False) and round_idx < len(args.CL_PHASE_EPOCHS):
            current_epochs = args.CL_PHASE_EPOCHS[round_idx]
            current_difficulty = args.CL_DIFFICULTY[round_idx]
            print(f"\nCurriculum Stage {round_idx+1}/{len(args.CL_PHASE_EPOCHS)}")
            print(f"Difficulty: {current_difficulty:.1f}, Epochs: {current_epochs}")
            
            algorithm.eval()
            
            transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
    
            # Create evaluator with transform capability
            class CurriculumEvaluator:
                def __init__(self, algorithm, transform_fn=None):
                    self.algorithm = algorithm
                    self.transform_fn = transform_fn
                    
                def eval(self):
                    self.algorithm.eval()
                    
                def predict(self, x):
                    # Apply normalization
                    x = (x - self.algorithm.data_mean) / self.algorithm.data_std
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
            algorithm.train()  # Ensure training mode
            for batch in train_loader:
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    # Apply normalization
                    inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    data = [inputs, labels, domains]
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    # Apply normalization
                    inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    data = [inputs, labels, domains]
                
                # Use safe cross entropy
                algorithm.criterion = safe_cross_entropy
                
                loss_result_dict = algorithm.update_a(data, opta)
                
                if 'class' not in loss_result_dict or not np.isfinite(loss_result_dict['class']):
                    print("Skipping batch due to NaN/inf loss in update_a")
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
            
            algorithm.train()  # Ensure training mode
            for batch in train_loader:
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    # Apply normalization
                    inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    data = [inputs, labels, domains]
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    # Apply normalization
                    inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    data = [inputs, labels, domains]
                
                loss_result_dict = algorithm.update_d(data, optd)
                
                if any(key not in loss_result_dict or not np.isfinite(loss_result_dict[key]) for key in ['total', 'dis', 'ent']):
                    print("Skipping batch due to NaN/inf loss in update_d")
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
            
            # Train for one epoch
            algorithm.train()  # Ensure training mode
            for batch_idx, batch in enumerate(train_loader):
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = batch[0].to(args.device)
                    labels = batch[1].to(args.device)
                    domains = batch[2].to(args.device)
                    # Apply normalization
                    inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    data = [inputs, labels, domains]
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    # Apply normalization
                    inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    data = [inputs, labels, domains]
                
                # Use safe cross entropy
                algorithm.criterion = safe_cross_entropy
                
                step_vals = algorithm.update(data, opt)
                torch.nn.utils.clip_grad_norm_(algorithm.parameters(), MAX_GRAD_NORM)
            
            # Evaluate after each epoch
            algorithm.eval()  # Ensure evaluation mode
            train_acc = evaluate_accuracy(train_loader_noshuffle)
            valid_acc = evaluate_accuracy(valid_loader)
            target_acc = evaluate_accuracy(target_loader)
            
            # Update learning rate
            scheduler.step(valid_acc)
            
            results = {
                'epoch': global_step,
                'train_acc': train_acc,
                'valid_acc': valid_acc,
                'target_acc': target_acc,
                'total_cost_time': time.time() - step_start_time
            }
            
            # Include the contrastive loss if available
            if 'contrast' in step_vals:
                results['contrast_loss'] = step_vals['contrast']
                logs['contrast_loss'].append(step_vals['contrast'])
            
            # Log classification and domain losses
            for key in ['class', 'dis']:
                if key in step_vals:
                    results[f"{key}_loss"] = step_vals[key]
                    logs[f"{key}_loss"].append(step_vals[key])
            
            for metric in ['train_acc', 'valid_acc', 'target_acc']:
                logs[metric].append(results[metric])
            
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
                epochs_without_improvement = 0
                torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))
            else:
                epochs_without_improvement += 1
                
            # Prepare row for printing
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
                        inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                        inputs = transform_for_gnn(inputs)
                    else:
                        inputs = data[0].to(args.device).float()
                        inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    
                    features = algorithm.featurizer(inputs).detach().cpu().numpy()
                    source_features.append(features)
            source_features = np.concatenate(source_features, axis=0)
            
            target_features = []
            with torch.no_grad():
                for data in target_loader:
                    if args.use_gnn and GNN_AVAILABLE:
                        inputs = data[0].to(args.device)
                        inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                        inputs = transform_for_gnn(inputs)
                    else:
                        inputs = data[0].to(args.device).float()
                        inputs = (inputs - algorithm.data_mean) / algorithm.data_std
                    
                    features = algorithm.featurizer(inputs).detach().cpu().numpy()
                    target_features.append(features)
            target_features = np.concatenate(target_features, axis=0)
            
            h_div, domain_acc = calculate_h_divergence(source_features, target_features)
            logs['h_divergence'].append(h_div)
            logs['domain_acc'].append(domain_acc)
            print(f" H-Divergence: {h_div:.4f}, Domain Classifier Acc: {domain_acc:.4f}")
            
            algorithm.train()
            
    print(f'\n🎯 Final Target Accuracy: {target_acc:.4f}')
    
    # ======================= SHAP EXPLAINABILITY =======================
    if getattr(args, 'enable_shap', False):
        print("\n📊 Running SHAP explainability...")
        try:
            # Prepare background and evaluation data
            if args.use_gnn and GNN_AVAILABLE:
                # Handle PyG DataBatch objects for GNN
                background_list = []
                for data in valid_loader:
                    background_list.append(data)
                    if len(background_list) * args.batch_size >= 64:
                        break
                background = background_list[0]  # Use first batch
                X_eval = background[:10]  # First 10 samples from the batch
            else:
                # Standard tensor handling for CNN
                background = get_background_batch(valid_loader, size=64).cuda()
                X_eval = background[:10]
            
            # Disable inplace operations in the model
            disable_inplace_relu(algorithm)
            
            # Create transform wrapper for GNN if needed
            transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
                
            # Transform background and X_eval if necessary
            if transform_fn is not None:
                background = transform_fn(background)
                X_eval = transform_fn(X_eval)
            
            # Apply normalization
            background = (background - algorithm.data_mean) / algorithm.data_std
            X_eval = (X_eval - algorithm.data_mean) / algorithm.data_std
            
            # Compute SHAP values safely
            shap_explanation = safe_compute_shap_values(algorithm, background, X_eval)
            
            # Extract values from Explanation object
            shap_vals = shap_explanation.values
            print(f"SHAP values shape: {shap_vals.shape}")
            
            # Convert to numpy safely before visualization
            X_eval_np = X_eval.detach().cpu().numpy()
            
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
                    print("Skipping visualization-specific reshaping")
            
            # Generate core visualizations
            try:
                plot_summary(shap_vals, X_eval_np, 
                            output_path=os.path.join(args.output, "shap_summary.png"))
            except IndexError as e:
                print(f"SHAP summary plot dimension error: {str(e)}")
                print(f"Using fallback 3D visualization instead")
                plot_emg_shap_4d(X_eval, shap_vals, 
                                output_path=os.path.join(args.output, "shap_3d_fallback.html"))
            
            overlay_signal_with_shap(X_eval_np[0], shap_vals, 
                                    output_path=os.path.join(args.output, "shap_overlay.png"))
            plot_shap_heatmap(shap_vals, 
                            output_path=os.path.join(args.output, "shap_heatmap.png"))
            
            # Evaluate SHAP impact
            base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals)
            
            # Save SHAP values
            save_path = os.path.join(args.output, "shap_values.npy")
            save_shap_numpy(shap_vals, save_path=save_path)
            
            # Compute impact metrics
            print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
            print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
            print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
            print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals):.4f}")
            
            # Compute advanced metrics
            metrics = evaluate_advanced_shap_metrics(shap_vals, X_eval)
            print(f"[SHAP] Entropy: {metrics.get('shap_entropy', 0):.4f}")
            print(f"[SHAP] Coherence: {metrics.get('feature_coherence', 0):.4f}")
            print(f"[SHAP] Channel Variance: {metrics.get('channel_variance', 0):.4f}")
            print(f"[SHAP] Temporal Entropy: {metrics.get('temporal_entropy', 0):.4f}")
            print(f"[SHAP] Mutual Info: {metrics.get('mutual_info', 0):.4f}")
            print(f"[SHAP] PCA Alignment: {metrics.get('pca_alignment', 0):.4f}")
            
            # Compute similarity metrics between first two samples
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
            
            # Generate 4D visualizations
            plot_emg_shap_4d(X_eval, shap_vals, 
                            output_path=os.path.join(args.output, "shap_4d_scatter.html"))
            plot_4d_shap_surface(shap_vals, 
                                output_path=os.path.join(args.output, "shap_4d_surface.html"))
            
            # Confusion matrix
            true_labels, pred_labels = [], []
            for data in valid_loader:
                if args.use_gnn and GNN_AVAILABLE:
                    x = data[0].to(args.device)
                    y = data[1]
                    # Apply normalization
                    x = (x - algorithm.data_mean) / algorithm.data_std
                    x = transform_for_gnn(x)
                else:
                    x = data[0].to(args.device).float()
                    y = data[1]
                    # Apply normalization
                    x = (x - algorithm.data_mean) / algorithm.data_std
                
                with torch.no_grad():
                    preds = algorithm.predict(x).cpu()
                    true_labels.extend(y.cpu().numpy())
                    pred_labels.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())
            
            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix (Validation Set)")
            plt.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=300)
            plt.close()
            
            print("✅ SHAP analysis completed successfully")
        except Exception as e:
            print(f"[ERROR] SHAP analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
    # ======================= END SHAP SECTION =======================
    
    try:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        epochs = list(range(len(logs['class_loss'])))
        plt.plot(epochs, logs['class_loss'], label="Class Loss", marker='o')
        plt.plot(epochs, logs['dis_loss'], label="Dis Loss", marker='x')
        plt.plot(epochs, logs['total_loss'], label="Total Loss", linestyle='--')
        plt.title("Losses over Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        epochs = list(range(len(logs['train_acc'])))
        plt.plot(epochs, logs['train_acc'], label="Train Accuracy", marker='o')
        plt.plot(epochs, logs['valid_acc'], label="Valid Accuracy", marker='x')
        plt.plot(epochs, logs['target_acc'], label="Target Accuracy", linestyle='--')
        plt.title("Accuracy over Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "training_metrics.png"), dpi=300)
        plt.close()
        print("✅ Training metrics plot saved")
        
        if logs['h_divergence']:
            plt.figure(figsize=(10, 6))
            h_epochs = [i * 5 for i in range(len(logs['h_divergence']))]
            plt.plot(h_epochs, logs['h_divergence'], 'o-', label='H-Divergence')
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
    # ====================== STABLE PARAMETER SETTINGS ======================
    args.lambda_cls = getattr(args, 'lambda_cls', 1.0)
    args.lambda_dis = getattr(args, 'lambda_dis', 0.001)  # Reduced for stability
    args.label_smoothing = 0.1  # Added label smoothing
    args.max_grad_norm = 1.0
    args.gnn_pretrain_epochs = getattr(args, 'gnn_pretrain_epochs', 5)
    
    # CRITICAL FIX: Add local_epoch parameter
    args.local_epoch = getattr(args, 'local_epoch', 5)

    # GNN parameters optimized
    if not hasattr(args, 'use_gnn'):
        args.use_gnn = False
        
    if args.use_gnn:
        if not GNN_AVAILABLE:
            print("[WARNING] GNN requested but not available. Falling back to CNN.")
            args.use_gnn = False
        else:
            args.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 128)
            args.gnn_output_dim = getattr(args, 'gnn_output_dim', 256)
            args.gnn_layers = 1
            args.gnn_lr = getattr(args, 'gnn_lr', 0.0001)
            args.gnn_weight_decay = 0.0
            
            args.use_tcn = getattr(args, 'use_tcn', True)
            args.lstm_hidden_size = 128
            args.lstm_layers = 1
            args.bidirectional = True
            args.lstm_dropout = 0.0

    # Optimizer settings optimized
    args.optimizer = getattr(args, 'optimizer', 'adam')
    args.weight_decay = 1e-4
    args.domain_adv_weight = 0.1
    args.lr = 0.0005  # Lower learning rate for stability

    # Augmentation enabled
    args.jitter_scale = 0.1
    args.scaling_std = 0.1
    args.warp_ratio = 0.1
    args.aug_prob = 0.5

    # Training schedule optimized
    args.max_epoch = getattr(args, 'max_epoch', 100)
    args.early_stopping_patience = 15  # Increased patience

    # Domain adaptation minimized
    if not hasattr(args, 'adv_weight'):
        args.adv_weight = 0.0

    main(args)
