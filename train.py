import os
import sys
import subprocess
import time
import torch
import torch.nn as nn
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
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from network.act_network import ActNetwork

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
        
        # Skip if only one cluster exists
        if len(np.unique(labels)) < 2:
            silhouette = -1
            dbi = float('inf')
            ch_score = -1
        else:
            silhouette = silhouette_score(features, labels)
            dbi = davies_bouldin_score(features, labels)
            ch_score = calinski_harabasz_score(features, labels)
        
        # Combine scores: higher silhouette and CH are better, lower DBI is better
        norm_silhouette = (silhouette + 1) / 2
        norm_dbi = 1 / (1 + dbi)
        norm_ch = ch_score / 1000
        
        # Combined score gives weight to all three metrics
        combined_score = (0.5 * norm_silhouette) + (0.3 * norm_ch) + (0.2 * norm_dbi)
        scores.append((k, silhouette, dbi, ch_score, combined_score))
        
        print(f"K={k}: Silhouette={silhouette:.4f}, DBI={dbi:.4f}, CH={ch_score:.4f}, Combined={combined_score:.4f}")
        
        if combined_score > best_score:
            best_k = k
            best_score = combined_score
            
    print(f"[INFO] Optimal K determined as {best_k} (Combined Score: {best_score:.4f})")
    return best_k

def calculate_h_divergence(features_source, features_target):
    """
    Calculate h-divergence between source and target domain features
    
    Args:
        features_source: Features from source domain (numpy array)
        features_target: Features from target domain (numpy array)
        
    Returns:
        h_divergence: Domain discrepancy measure
        domain_acc: Domain classifier accuracy
    """
    # Create domain labels: 0 for source, 1 for target
    labels_source = np.zeros(features_source.shape[0])
    labels_target = np.ones(features_target.shape[0])
    
    # Combine features and labels
    X = np.vstack([features_source, features_target])
    y = np.hstack([labels_source, labels_target])
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train and test sets (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train a simple domain classifier
    domain_classifier = LogisticRegression(max_iter=1000, random_state=42)
    domain_classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    domain_acc = domain_classifier.score(X_test, y_test)
    
    # Calculate h-divergence: d = 2(1 - 2ε)
    h_divergence = 2 * (1 - 2 * (1 - domain_acc))
    
    return h_divergence, domain_acc

def transform_for_gnn(x):
    """Transform 4D input to 3D for GNN models"""
    if x.dim() == 4 and x.shape[2] == 1:
        return x.squeeze(2).permute(0, 2, 1)
    return x

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)
    print_environ()
    print(s)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load datasets
    loader_data = get_act_dataloader(args)
    train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata = loader_data[:7]
    
    # Automated K estimation if enabled
    if getattr(args, 'automated_k', False):
        print("\nRunning automated K estimation...")
        
        # Use GNN if enabled, otherwise use standard CNN
        if args.use_gnn and GNN_AVAILABLE:
            print("Using GNN for feature extraction")
            # Initialize graph builder for feature extraction
            graph_builder = GraphBuilder(
                method='correlation',
                threshold_type='adaptive',
                default_threshold=0.3,
                adaptive_factor=1.5
            )
            temp_model = TemporalGCN(
                input_dim=8,  # EMG channels
                hidden_dim=args.gnn_hidden_dim,
                output_dim=args.gnn_output_dim,
                graph_builder=graph_builder
            ).cuda()
        else:
            temp_model = ActNetwork(args.dataset).cuda()
            
        temp_model.eval()
        feature_list = []
        
        with torch.no_grad():
            for batch in train_loader:
                inputs = batch[0].cuda().float()
                
                # Handle GNN input format if needed
                if args.use_gnn and GNN_AVAILABLE:
                    inputs = inputs.squeeze(2).permute(0, 2, 1)  # Convert to (batch, time, features)
                
                features = temp_model(inputs)
                feature_list.append(features.cpu().numpy())
                
        all_features = np.concatenate(feature_list, axis=0)
        optimal_k = automated_k_estimation(all_features)
        args.latent_domain_num = optimal_k
        print(f"Using automated latent_domain_num (K): {args.latent_domain_num}")
        
        del temp_model
        
        # Batch size adjustment
        if args.latent_domain_num < 6:
            args.batch_size = 32 * args.latent_domain_num
        else:
            args.batch_size = 16 * args.latent_domain_num
        print(f"Adjusted batch size: {args.batch_size}")
        
        # Recreate data loaders with new batch size
        train_loader = DataLoader(
            dataset=tr,
            batch_size=args.batch_size,
            num_workers=min(2, args.N_WORKERS),
            drop_last=False,
            shuffle=True
        )
        train_loader_noshuffle = DataLoader(
            dataset=tr,
            batch_size=args.batch_size,
            num_workers=min(2, args.N_WORKERS),
            drop_last=False,
            shuffle=False
        )
        valid_loader = DataLoader(
            dataset=val,
            batch_size=args.batch_size,
            num_workers=min(2, args.N_WORKERS),
            drop_last=False,
            shuffle=False
        )
        target_loader = DataLoader(
            dataset=targetdata,
            batch_size=args.batch_size,
            num_workers=min(2, args.N_WORKERS),
            drop_last=False,
            shuffle=False
        )
    
    # Initialize algorithm
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    
    # ======================= GNN INITIALIZATION START =======================
    if args.use_gnn and GNN_AVAILABLE:
        print("\n===== Initializing GNN Feature Extractor =====")
        
        # Initialize graph builder with research-optimized parameters
        graph_builder = GraphBuilder(
            method='correlation',
            threshold_type='adaptive',
            default_threshold=0.3,
            adaptive_factor=1.5,
            fully_connected_fallback=True
        )
        
        # FIXED: Enhanced Temporal GCN with proper skip connection
        class EnhancedTemporalGCN(TemporalGCN):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Add skip connection with temporal aggregation
                if kwargs['input_dim'] != kwargs['output_dim']:
                    self.skip_conn = nn.Sequential(
                        nn.Linear(kwargs['input_dim'], kwargs['output_dim']),
                        nn.ReLU(),
                    )
                    self.temporal_aggregator = nn.AdaptiveAvgPool1d(1)
                else:
                    self.skip_conn = nn.Identity()
                    self.temporal_aggregator = nn.Identity()
                    
            def forward(self, x):
                # Original processing
                out = super().forward(x)
                # Process skip connection with temporal aggregation
                skip_out = self.skip_conn(x)  # [batch, time, output_dim]
                skip_out = skip_out.permute(0, 2, 1)  # [batch, output_dim, time]
                skip_out = self.temporal_aggregator(skip_out)  # [batch, output_dim, 1]
                skip_out = skip_out.squeeze(2)  # [batch, output_dim]
                return out + skip_out
        
        gnn_model = EnhancedTemporalGCN(
            input_dim=8,  # EMG channels
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            graph_builder=graph_builder
        ).cuda()
        
        # Replace CNN feature extractor with GNN
        algorithm.featurizer = gnn_model
        
        # Create a function to build consistent bottleneck layers
        def create_bottleneck(input_dim, output_dim, layer_spec):
            """Create a bottleneck layer with consistent architecture"""
            try:
                num_layers = int(layer_spec)
                layers = []
                current_dim = input_dim
                
                # Add intermediate layers
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(current_dim, current_dim))
                    layers.append(nn.BatchNorm1d(current_dim))
                    layers.append(nn.ReLU(inplace=True))
                
                # Add final projection layer
                layers.append(nn.Linear(current_dim, output_dim))
                return nn.Sequential(*layers)
            
            except ValueError:
                # Fallback to simple linear projection
                return nn.Sequential(nn.Linear(input_dim, output_dim))
        
        # Create both bottlenecks with the correct dimensions
        input_dim = args.gnn_output_dim
        output_dim = int(args.bottleneck)
        
        # Create both bottlenecks (classifier and adversarial)
        algorithm.bottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        algorithm.abottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        algorithm.dbottleneck = create_bottleneck(input_dim, output_dim, args.layer).cuda()
        print(f"Created bottlenecks: {input_dim} -> {output_dim}")
        print(f"Bottleneck architecture: {algorithm.bottleneck}")
        
        # GNN Pretraining if enabled
        if hasattr(args, 'gnn_pretrain_epochs') and args.gnn_pretrain_epochs > 0:
            print(f"\n==== GNN Pretraining ({args.gnn_pretrain_epochs} epochs) ====")
            gnn_optimizer = torch.optim.Adam(
                gnn_model.parameters(),
                lr=args.gnn_lr,
                weight_decay=args.gnn_weight_decay
            )
            
            for epoch in range(args.gnn_pretrain_epochs):
                gnn_model.train()
                total_loss = 0
                for batch in train_loader:
                    x = batch[0].cuda().float()
                    
                    # Convert to (batch, time, features) format
                    if args.use_gnn and GNN_AVAILABLE:
                        x = transform_for_gnn(x)
                        if x.dim() != 3:
                            raise ValueError(f"GNN requires 3D input (B,T,C), got {x.shape}")
                    
                    # Calculate mean across time dimension
                    target = torch.mean(x, dim=1)  # [batch, channels]        
                    
                    # Forward pass
                    features = gnn_model(x)
                    
                    # Reconstruction loss
                    reconstructed = gnn_model.reconstruct(features)
                    loss = torch.nn.functional.mse_loss(reconstructed, target)
                    
                    # Skip update if NaN loss
                    if torch.isnan(loss):
                        print("NaN loss detected during pretraining, skipping update")
                        continue
                    
                    # Optimization with gradient clipping
                    gnn_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
                    gnn_optimizer.step()
                    
                    total_loss += loss.item()
                    
                print(f'GNN Pretrain Epoch {epoch+1}/{args.gnn_pretrain_epochs}: Loss {total_loss/len(train_loader):.4f}')
            
            print("GNN pretraining complete")
    # ======================= GNN INITIALIZATION END =======================
    
    algorithm.train()
    
    # Setup optimizers
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training metrics logging
    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'ent_loss',
                           'total_loss', 'train_acc', 'valid_acc', 'target_acc',
                           'total_cost_time', 'h_divergence', 'domain_acc']}
    best_valid_acc, target_acc = 0, 0
    
    # Create entire source loader for h-divergence calculation
    entire_source_loader = DataLoader(
        tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.N_WORKERS))
    
    # Main training loop
    global_step = 0
    for round_idx in range(args.max_epoch):
        print(f'\n======== ROUND {round_idx} ========')
        
        # Determine epochs for this round
        if getattr(args, 'curriculum', False) and round_idx < getattr(args, 'CL_PHASE_EPOCHS', 5):
            current_epochs = args.CL_PHASE_EPOCHS
            print(f"Curriculum learning: Stage {round_idx} (using {current_epochs} epochs)")
        else:
            current_epochs = args.local_epoch
        
               # Curriculum learning setup
        if getattr(args, 'curriculum', False) and round_idx < getattr(args, 'CL_PHASE_EPOCHS', 5):
            print(f"Curriculum learning: Stage {round_idx}")
            
            # Create a wrapped predict function for domain evaluation
            def wrapped_predict(model, x):
                if args.use_gnn and GNN_AVAILABLE:
                    x = transform_for_gnn(x)
                return model.predict(x)
            
            # Create a temporary algorithm with wrapped predict
            class TempAlgorithmWrapper:
                def __init__(self, model):
                    self.model = model
                
                def predict(self, x):
                    return wrapped_predict(self.model, x)
            
            temp_algorithm = TempAlgorithmWrapper(algorithm)
            
            # Get the curriculum loader using wrapped predict
            train_loader = get_curriculum_loader(
                args,
                temp_algorithm,
                tr,
                val,
                stage=round_idx
            )
            
            # Apply GNN transformation to the final training loader
            if args.use_gnn and GNN_AVAILABLE:
                # Create transformed dataset
                class TransformedDataset(torch.utils.data.Dataset):
                    def __init__(self, original_dataset):
                        self.original_dataset = original_dataset
                        
                    def __len__(self):
                        return len(self.original_dataset)
                    
                    def __getitem__(self, idx):
                        data = self.original_dataset[idx]
                        if len(data) == 3:  # (input, class_label, domain_label)
                            x, y, d = data
                            x = transform_for_gnn(x)
                            return x, y, d
                        elif len(data) == 2:  # (input, class_label)
                            x, y = data
                            x = transform_for_gnn(x)
                            return x, y
                        else:
                            x = data[0]
                            x = transform_for_gnn(x)
                            return (x, *data[1:])
                
                # Wrap the dataset with transformation
                transformed_dataset = TransformedDataset(train_loader.dataset)
                train_loader = DataLoader(
                    transformed_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=min(2, args.N_WORKERS)
                )
            
            # Update the no-shuffle loader as well
            train_loader_noshuffle = DataLoader(
                train_loader.dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=min(2, args.N_WORKERS)
            )
        
        algorithm.train()
        
        # Phase 1: Feature update
        print('\n==== Feature update ====')
        print_row(['epoch', 'class_loss'], colwidth=15)
        for step in range(current_epochs):
            epoch_class_loss = 0.0
            batch_count = 0
            
            for data in train_loader:
                # GNN input transformation
                if args.use_gnn and GNN_AVAILABLE:
                    data = list(data)
                    data[0] = transform_for_gnn(data[0])
                
                loss_result_dict = algorithm.update_a(data, opta)
                
                # Skip update if NaN loss
                if not np.isfinite(loss_result_dict['class']):
                    print("Skipping step due to non-finite loss")
                    continue
                    
                epoch_class_loss += loss_result_dict['class']
                batch_count += 1
            
            if batch_count > 0:
                epoch_class_loss /= batch_count
                print_row([step, epoch_class_loss], colwidth=15)
                logs['class_loss'].append(epoch_class_loss)
        
        # Phase 2: Latent domain characterization
        print('\n==== Latent domain characterization ====')
        print_row(['epoch', 'total_loss', 'dis_loss', 'ent_loss'], colwidth=15)
        for step in range(current_epochs):
            epoch_total = 0.0
            epoch_dis = 0.0
            epoch_ent = 0.0
            batch_count = 0
            
            for data in train_loader:
                # GNN input transformation
                if args.use_gnn and GNN_AVAILABLE:
                    data = list(data)
                    data[0] = transform_for_gnn(data[0])
                
                loss_result_dict = algorithm.update_d(data, optd)
                
                if any(not np.isfinite(v) for v in loss_result_dict.values()):
                    print("Skipping step due to non-finite loss")
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
            for data in train_loader:
                # GNN input transformation
                if args.use_gnn and GNN_AVAILABLE:
                    data = list(data)
                    data[0] = transform_for_gnn(data[0])
                
                step_vals = algorithm.update(data, opt)
                
                # Apply gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(algorithm.parameters(), 1.0)
            
            # Create transform wrapper for GNN if needed
            transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
                
            # Calculate accuracies
            results = {
                'epoch': global_step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None, transform_fn=transform_fn),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None, transform_fn=transform_fn),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None, transform_fn=transform_fn),
                'total_cost_time': time.time() - step_start_time
            }
            
            # Update scheduler
            if scheduler:
                scheduler.step(results['valid_acc'])
            
            # Log losses
            for key in loss_list:
                results[f"{key}_loss"] = step_vals[key]
                logs[f"{key}_loss"].append(step_vals[key])
            
            # Log metrics
            for metric in ['train_acc', 'valid_acc', 'target_acc']:
                logs[metric].append(results[metric])
            
            # Update best validation accuracy
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            
            print_row([results[key] for key in print_key], colwidth=15)
            global_step += 1
        
        logs['total_cost_time'].append(time.time() - round_start_time)
        
        # Calculate h-divergence every 5 epochs
        if round_idx % 5 == 0:
            print("\nCalculating h-divergence...")
            algorithm.eval()
            
            # Extract features for source domain
            source_features = []
            with torch.no_grad():
                for data in entire_source_loader:
                    x = data[0].cuda().float()
                    
                    # GNN input transformation
                    if args.use_gnn and GNN_AVAILABLE:
                        x = transform_for_gnn(x)
                    
                    features = algorithm.featurizer(x).detach().cpu().numpy()
                    source_features.append(features)
            source_features = np.concatenate(source_features, axis=0)
            
            # Extract features for target domain
            target_features = []
            with torch.no_grad():
                for data in target_loader:
                    x = data[0].cuda().float()
                    
                    # GNN input transformation
                    if args.use_gnn and GNN_AVAILABLE:
                        x = transform_for_gnn(x)
                    
                    features = algorithm.featurizer(x).detach().cpu().numpy()
                    target_features.append(features)
            target_features = np.concatenate(target_features, axis=0)
            
            # Calculate h-divergence
            h_div, domain_acc = calculate_h_divergence(source_features, target_features)
            logs['h_divergence'].append(h_div)
            logs['domain_acc'].append(domain_acc)
            print(f" H-Divergence: {h_div:.4f}, Domain Classifier Acc: {domain_acc:.4f}")
            
            algorithm.train()
    
    print(f'\n🎯 Final Target Accuracy: {target_acc:.4f}')
    
    # SHAP explainability analysis
    if getattr(args, 'enable_shap', False):
        print("\n📊 Running SHAP explainability...")
        try:
            # Prepare background and evaluation data
            background = get_background_batch(valid_loader, size=64).cuda()
            X_eval = background[:10]
            
            # Disable inplace operations in the model
            disable_inplace_relu(algorithm)
            
            # Create transform wrapper for GNN if needed
            transform_fn = transform_for_gnn if args.use_gnn and GNN_AVAILABLE else None
                
            # Compute SHAP values safely
            shap_vals = safe_compute_shap_values(algorithm, background, X_eval, transform_fn=transform_fn)
            
            # Convert to numpy safely before visualization
            X_eval_np = X_eval.detach().cpu().numpy()
            
            # Handle GNN dimensionality for visualization
            if args.use_gnn and GNN_AVAILABLE:
                # Convert 3D (batch, time, channels) to 4D (batch, channels, 1, time)
                shap_vals = np.transpose(shap_vals, (0, 2, 1))
                shap_vals = np.expand_dims(shap_vals, axis=2)
                
                X_eval_np = np.transpose(X_eval_np, (0, 2, 1))
                X_eval_np = np.expand_dims(X_eval_np, axis=2)
            
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
            base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals, transform_fn=transform_fn)
            
            # Save SHAP values
            save_path = os.path.join(args.output, "shap_values.npy")
            save_shap_numpy(shap_vals, save_path=save_path)
            
            # Compute impact metrics
            print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
            print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
            print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
            print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals, transform_fn=transform_fn):.4f}")
            
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
                x, y = data[0].cuda(), data[1]
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
    
    # Plot training metrics
    try:
        # Main training metrics plot
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
        
        # H-Divergence plot
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
    
    # Add GNN-specific parameters to args
    if not hasattr(args, 'use_gnn'):
        args.use_gnn = False
        
    if args.use_gnn:
        if not GNN_AVAILABLE:
            print("[WARNING] GNN requested but not available. Falling back to CNN.")
            args.use_gnn = False
        else:
            # GNN hyperparameters
            args.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 32)
            args.gnn_output_dim = getattr(args, 'gnn_output_dim', 128)
            args.gnn_lr = getattr(args, 'gnn_lr', 0.001)
            args.gnn_weight_decay = getattr(args, 'gnn_weight_decay', 0.0001)
            args.gnn_pretrain_epochs = getattr(args, 'gnn_pretrain_epochs', 5)
    
    # Increase adversarial weight for better domain adaptation
    if not hasattr(args, 'adv_weight'):
        args.adv_weight = 2.0
    
    main(args)
