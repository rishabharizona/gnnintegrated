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
    GNN_AVAILABLE = True
    print("GNN modules successfully imported")
except ImportError as e:
    print(f"[WARNING] GNN modules not available: {str(e)}")
    print("Falling back to CNN architecture")
    GNN_AVAILABLE = False
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
        self.jitter_scale = 0.0  # MINIMAL
        self.scaling_std = 0.0  # MINIMAL
        self.warp_ratio = 0.0  # MINIMAL
        self.aug_prob = 0.0  # MINIMAL

    def forward(self, x):
        return x  # COMPLETELY DISABLED

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
    def __init__(self, *args, **kwargs):
        self.n_layers = kwargs.pop('n_layers', 1)  # MINIMAL LAYERS
        self.use_tcn = kwargs.pop('use_tcn', False)
        
        lstm_hidden_size = kwargs.pop('lstm_hidden_size', 64)  # REDUCED
        lstm_layers = kwargs.pop('lstm_layers', 1)
        bidirectional = kwargs.pop('bidirectional', False)
        
        super().__init__(*args, **kwargs)
        
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
            num_heads=2,  # REDUCED HEADS
            dropout=0.0,  # ZERO DROPOUT
            batch_first=True
        )
        
        if self.use_tcn:
            tcn_layers = []
            num_channels = [self.hidden_dim] * 2  # REDUCED LAYERS
            kernel_size = 3  # REDUCED
            for i in range(len(num_channels)):
                dilation = 2 ** i
                in_channels = self.hidden_dim if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
                tcn_layers += [TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation, dropout=0.0  # ZERO DROPOUT
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
                dropout=0.0  # ZERO DROPOUT
            )
            lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
            self.lstm_proj = nn.Linear(lstm_output_dim, self.output_dim)
            self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        
        self.temporal_norm = nn.LayerNorm(self.output_dim)
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),  # Changed from ReLU to ReLU (no change needed)
            nn.Linear(self.output_dim, self.output_dim)
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

    def forward(self, x):
        if hasattr(x, 'x') and hasattr(x, 'batch'):
            from torch_geometric.utils import to_dense_batch
            x, mask = to_dense_batch(x.x, x.batch)
        
        if x.dim() == 4:
            x = x.squeeze(2).permute(0, 2, 1)
        
        if x.size(-1) not in [8, 200]:
            raise ValueError(f"Input features dim mismatch! Expected 8 or 200, got {x.size(-1)}")
        
        if x.size(-1) == 200 and self.input_dim == 8:
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Linear(200, 8).to(x.device)
            x = self.feature_projection(x)
        
        original_x = x.clone()
        
        gnn_features = x
        for layer, norm in zip(self.gnn_layers, self.norms):
            gnn_features = layer(gnn_features)
            gnn_features = norm(gnn_features)
            gnn_features = F.relu(gnn_features)  # Changed from gelu to relu
        
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
    
    # Handle curriculum parameters
    if getattr(args, 'curriculum', False):
        # Handle CL_PHASE_EPOCHS: if it's an integer, convert to a list of that integer for 3 phases
        if hasattr(args, 'CL_PHASE_EPOCHS'):
            if isinstance(args.CL_PHASE_EPOCHS, int):
                args.CL_PHASE_EPOCHS = [args.CL_PHASE_EPOCHS] * 3
            elif not isinstance(args.CL_PHASE_EPOCHS, list):
                args.CL_PHASE_EPOCHS = [3, 3, 3]  # default
        else:
            args.CL_PHASE_EPOCHS = [3, 3, 3]
        
        # Similarly for CL_DIFFICULTY
        if not hasattr(args, 'CL_DIFFICULTY') or not isinstance(args.CL_DIFFICULTY, list):
            args.CL_DIFFICULTY = [0.2, 0.5, 0.8]  # default
        
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
        num_workers=min(2, args.N_WORKERS))  # REDUCED WORKERS
    
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
                n_layers=getattr(args, 'gnn_layers', 1),  # MINIMAL LAYERS
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
                    x = transform_for_gnn(inputs)
                else:
                    inputs = batch[0].to(args.device).float()
                    labels = batch[1].to(args.device).long()
                    domains = batch[2].to(args.device).long()
                    x = inputs
                
                features = temp_model(x)
                feature_list.append(features.detach().cpu().numpy())
        
        all_features = np.concatenate(feature_list, axis=0)
        optimal_k = automated_k_estimation(all_features)
        args.latent_domain_num = optimal_k
        print(f"Using automated latent_domain_num (K): {args.latent_domain_num}")
        
        del temp_model
        torch.cuda.empty_cache()
    
    if args.latent_domain_num < 6:
        args.batch_size = 16 * args.latent_domain_num  # REDUCED
    else:
        args.batch_size = 8 * args.latent_domain_num  # REDUCED
    print(f"Adjusted batch size: {args.batch_size}")
    
    train_loader = LoaderClass(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=min(2, args.N_WORKERS),  # REDUCED
        drop_last=False,
        shuffle=True
    )
    
    train_loader_noshuffle = LoaderClass(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=min(2, args.N_WORKERS),  # REDUCED
        drop_last=False,
        shuffle=False
    )
    
    valid_loader = LoaderClass(
        dataset=val,
        batch_size=args.batch_size,
        num_workers=min(2, args.N_WORKERS),  # REDUCED
        drop_last=False,
        shuffle=False
    )
    
    target_loader = LoaderClass(
        dataset=targetdata,
        batch_size=args.batch_size,
        num_workers=min(2, args.N_WORKERS),  # REDUCED
        drop_last=False,
        shuffle=False
    )
    
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(args.device)
    
    if args.use_gnn and GNN_AVAILABLE:
        print("\n===== Initializing GNN Feature Extractor =====")
        
        graph_builder = GraphBuilder(
            method='correlation',
            threshold_type='adaptive',
            default_threshold=0.3,
            adaptive_factor=1.5,
            fully_connected_fallback=True
        )
        
        args.gnn_layers = getattr(args, 'gnn_layers', 1)  # MINIMAL LAYERS
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
        
        def create_bottleneck(input_dim, output_dim, layer_spec):
            try:
                num_layers = int(layer_spec)
                layers = []
                current_dim = input_dim
                
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(current_dim, current_dim))
                    layers.append(nn.BatchNorm1d(current_dim))
                    layers.append(nn.ReLU(inplace=True))
                
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
        
        # Enhanced GNN pretraining
        if hasattr(args, 'gnn_pretrain_epochs') and args.gnn_pretrain_epochs > 0:
            print(f"\n==== GNN Pretraining ({args.gnn_pretrain_epochs} epochs) ====")
            gnn_optimizer = torch.optim.Adam(
                algorithm.featurizer.parameters(),
                lr=args.gnn_lr,
                weight_decay=0.0
            )
            
            for epoch in range(args.gnn_pretrain_epochs):
                gnn_model.train()
                total_loss = 0
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
                        # Convert to [batch, channels, time] and then average
                        x_processed = x.squeeze(2).permute(0, 2, 1)
                        target = torch.mean(x_processed, dim=1)
                    else:
                        target = torch.mean(x, dim=1)

                    features = gnn_model(x)

                    # For reconstruction, we need to match the target shape
                    # Add reconstruction head if not exists
                    if not hasattr(gnn_model, 'reconstruction_head'):
                        # Create a reconstruction head that matches the target dimension
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
    
    algorithm.train()
    
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')
    
    augmenter = EMGDataAugmentation(args).cuda()
    
    if getattr(args, 'domain_adv_weight', 0.0) > 0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(
            bottleneck_dim=int(args.bottleneck)
        ).cuda()
        print(f"Added domain adversarial training (weight: {args.domain_adv_weight})")
    
    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'ent_loss',
                            'total_loss', 'train_acc', 'valid_acc', 'target_acc',
                            'total_cost_time', 'h_divergence', 'domain_acc']}
    best_valid_acc, target_acc = 0, 0
    
    entire_source_loader = LoaderClass(
        tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.N_WORKERS))  # REDUCED
    
    best_valid_acc = 0
    epochs_without_improvement = 0
    early_stopping_patience = getattr(args, 'early_stopping_patience', 20)  # INCREASED
    
    MAX_GRAD_NORM = 1.0  # Loosened gradient clipping
    
    global_step = 0
    for round_idx in range(args.max_epoch):
        if hasattr(algorithm.featurizer, 'dropout'):
            algorithm.featurizer.dropout.p = 0.0  # ZERO DROPOUT
        
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
            
            if hasattr(algorithm, 'eval'):
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
        print_key = ['epoch'] + [f"{item}_loss" for item in loss_list] + \
                   [f"{item}_acc" for item in eval_dict] + ['total_cost_time']
        print_row(print_key, colwidth=15)
        
        round_start_time = time.time()
        for step in range(current_epochs):
            step_start_time = time.time()
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
                
                step_vals = algorithm.update(data, opt)
                torch.nn.utils.clip_grad_norm_(algorithm.parameters(), MAX_GRAD_NORM)
            
            transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
            def evaluate_accuracy(loader):
                correct = 0
                total = 0
                algorithm.eval()
                with torch.no_grad():
                    for batch in loader:
                        if args.use_gnn and GNN_AVAILABLE:
                            inputs = batch[0].to(args.device)
                            labels = batch[1].to(args.device)
                            domains = batch[2].to(args.device)
                            if transform_fn:
                                inputs = transform_fn(inputs)
                        else:
                            inputs = batch[0].to(args.device).float()
                            labels = batch[1].to(args.device).long()
                            domains = batch[2].to(args.device).long()
                        
                        outputs = algorithm.predict(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                algorithm.train()
                return 100 * correct / total
            results = {
                'epoch': global_step,
                'train_acc': evaluate_accuracy(train_loader_noshuffle),
                'valid_acc': evaluate_accuracy(valid_loader),
                'target_acc': evaluate_accuracy(target_loader),
                'total_cost_time': time.time() - step_start_time
            }
            
            for key in loss_list:
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
                
            print_row([results[key] for key in print_key], colwidth=15)
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
                else:
                    x = data[0].to(args.device).float()
                    y = data[1]
                
                with torch.no_grad():
                    # Apply transform for GNN if needed
                    if args.use_gnn and GNN_AVAILABLE:
                        x = transform_for_gnn(x)
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
    # ====================== MINIMAL PARAMETER SETTINGS ======================
    # All regularization removed
    args.lambda_cls = getattr(args, 'lambda_cls', 1.0)
    args.lambda_dis = getattr(args, 'lambda_dis', 0.01)  # MINIMAL
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
            args.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 64)  # Increased
            args.gnn_output_dim = getattr(args, 'gnn_output_dim', 128)  # Increased
            args.gnn_layers = 1  # MINIMAL LAYERS
            args.gnn_lr = getattr(args, 'gnn_lr', 0.00005)  # MINIMAL
            args.gnn_weight_decay = 0.0  # DISABLED
            
            args.use_tcn = getattr(args, 'use_tcn', True)
            args.lstm_hidden_size = 64  # Increased
            args.lstm_layers = 1
            args.bidirectional = False  # DISABLED
            args.lstm_dropout = 0.0  # DISABLED

    # Optimizer settings minimized
    args.optimizer = getattr(args, 'optimizer', 'adam')
    args.weight_decay = 1e-4  # Enable weight decay
    args.domain_adv_weight = 0.1  # Enable domain adaptation
    args.lr = 0.0001  # Increased learning rate

    # Augmentation completely disabled
    args.jitter_scale = 0.0
    args.scaling_std = 0.0
    args.warp_ratio = 0.0
    args.channel_dropout = 0.0
    args.aug_prob = 0.0

    # Training schedule minimized
    args.max_epoch = getattr(args, 'max_epoch', 50)  # REDUCED
    args.early_stopping_patience = 20  # INCREASED

    # Domain adaptation minimized
    if not hasattr(args, 'adv_weight'):
        args.adv_weight = 0.0  # DISABLED

    main(args)
