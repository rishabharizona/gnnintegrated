import os
import sys
import subprocess

# Fix environment at the very top
try:
    # Create a compatible environment
    install_cmds = [
        # Remove conflicting packages
        [sys.executable, "-m", "pip", "uninstall", "-y", "thinc", "spacy"],
        
        # Install compatible NumPy version
        [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--quiet"],
        
        # Install PyTorch stack
        [sys.executable, "-m", "pip", "install", 
         "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
         "--index-url", "https://download.pytorch.org/whl/cu117", "--quiet"],
         
        # Install PyG packages
        [sys.executable, "-m", "pip", "install", "--force-reinstall",
         "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv",
         "-f", "https://data.pyg.org/whl/torch-2.0.1+cu117.html", "--quiet"],
         
        [sys.executable, "-m", "pip", "install", 
         "torch-geometric", "-f", "https://data.pyg.org/whl/torch-2.0.1+cu117.html", "--quiet"]
    ]
    
    for cmd in install_cmds:
        subprocess.check_call(cmd)
    
    # Restart to apply changes
    os.execv(sys.executable, [sys.executable] + sys.argv)
    
except Exception as e:
    print(f"Environment setup error: {e}")
    sys.exit(1)
import time
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression  # For h-divergence
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ, disable_inplace_relu
from datautil.getdataloader_single import get_act_dataloader, get_curriculum_loader
from torch.utils.data import DataLoader, ConcatDataset
from network.act_network import ActNetwork
from sklearn.metrics import davies_bouldin_score
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
# Add this class definition at the top of your file
class SubsetWithLabelSetter(torch.utils.data.Subset):
    """Subset that allows setting domain labels"""
    def __init__(self, dataset, indices, domain_label=None):
        super().__init__(dataset, indices)
        self.domain_label = domain_label
        
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        if self.domain_label is not None:
            # Return (x, y, new_domain_label) instead of original domain
            return (data[0], data[1], self.domain_label)
        return data

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
        else:
            silhouette = silhouette_score(features, labels)
            dbi = davies_bouldin_score(features, labels)
        
        # Combine scores: higher silhouette is better, lower DBI is better
        # Normalize and combine (silhouette in [-1,1], DBI in [0,inf])
        norm_silhouette = (silhouette + 1) / 2  # Map to [0,1]
        norm_dbi = 1 / (1 + dbi)  # Map to (0,1] where higher is better
        
        # Combined score gives equal weight to both metrics
        combined_score = (norm_silhouette + norm_dbi) / 2
        scores.append((k, silhouette, dbi, combined_score))
        
        print(f"K={k}: Silhouette={silhouette:.4f}, DBI={dbi:.4f}, Combined={combined_score:.4f}")
        
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
    # Where ε is the error rate of the domain classifier
    h_divergence = 2 * (1 - 2 * (1 - domain_acc))
    return h_divergence, domain_acc

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
        print("Running automated K estimation...")
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
        num_workers=args.N_WORKERS, 
        drop_last=False, 
        shuffle=True
    )
    
    train_loader_noshuffle = DataLoader(
        dataset=tr, 
        batch_size=args.batch_size, 
        num_workers=args.N_WORKERS, 
        drop_last=False, 
        shuffle=False
    )
    
    valid_loader = DataLoader(
        dataset=val, 
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS, 
        drop_last=False, 
        shuffle=False
    )
    
    target_loader = DataLoader(
        dataset=targetdata, 
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS, 
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
        
        # Initialize Temporal GCN model
        gnn_model = TemporalGCN(
            input_dim=8,  # EMG channels
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            graph_builder=graph_builder
        ).cuda()
        
        # Replace CNN feature extractor with GNN
        algorithm.featurizer = gnn_model
        print(f"GNN initialized: hidden_dim={args.gnn_hidden_dim}, output_dim={args.gnn_output_dim}")
        
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
                    x = x.squeeze(2).permute(0, 2, 1)
                    
                    # Forward pass
                    features = gnn_model(x)
                    
                    # Reconstruction loss
                    reconstructed = gnn_model.reconstruct(features)
                    loss = torch.nn.functional.mse_loss(reconstructed, x)
                    
                    # Optimization
                    gnn_optimizer.zero_grad()
                    loss.backward()
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
        num_workers=args.N_WORKERS
    )

    # Main training loop
    global_step = 0  # Initialize global step counter
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
            
            # Use advanced domain-based curriculum loader
            train_loader = get_curriculum_loader(
                args, 
                algorithm, 
                tr, 
                val, 
                stage=round_idx
            )
            
            # Update the no-shuffle loader as well (optional)
            train_loader_noshuffle = DataLoader(
                train_loader.dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=args.N_WORKERS
            )
            
            algorithm.train()
    
        # Phase 1: Feature update
        print('==== Feature update ====')
        print_row(['epoch', 'class_loss'], colwidth=15)
        for step in range(current_epochs):
            for data in train_loader:
                # ======================= GNN INPUT TRANSFORMATION =======================
                if args.use_gnn and GNN_AVAILABLE:
                    data = list(data)
                    # Convert from (batch, channels, 1, time) to (batch, time, channels)
                    data[0] = data[0].squeeze(2).permute(0, 2, 1)
                # ========================================================================
                
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step, loss_result_dict['class']], colwidth=15)
            logs['class_loss'].append(loss_result_dict['class'])
    
        # Phase 2: Latent domain characterization
        print('==== Latent domain characterization ====')
        print_row(['epoch', 'total_loss', 'dis_loss', 'ent_loss'], colwidth=15)
        for step in range(current_epochs):
            for data in train_loader:
                # ======================= GNN INPUT TRANSFORMATION =======================
                if args.use_gnn and GNN_AVAILABLE:
                    data = list(data)
                    data[0] = data[0].squeeze(2).permute(0, 2, 1)
                # ========================================================================
                
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step, loss_result_dict['total'], loss_result_dict['dis'], loss_result_dict['ent']], colwidth=15)
            logs['dis_loss'].append(loss_result_dict['dis'])
            logs['ent_loss'].append(loss_result_dict['ent'])
            logs['total_loss'].append(loss_result_dict['total'])
    
        algorithm.set_dlabel(train_loader)
    
        # Phase 3: Domain-invariant learning
        print('==== Domain-invariant feature learning ====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch'] + [f"{item}_loss" for item in loss_list] + \
                   [f"{item}_acc" for item in eval_dict] + ['total_cost_time']
        print_row(print_key, colwidth=15)
    
        round_start_time = time.time()
        for step in range(current_epochs):
            step_start_time = time.time()
            for data in train_loader:
                # ======================= GNN INPUT TRANSFORMATION =======================
                if args.use_gnn and GNN_AVAILABLE:
                    data = list(data)
                    data[0] = data[0].squeeze(2).permute(0, 2, 1)
                # ========================================================================
                
                step_vals = algorithm.update(data, opt)
    
            # Calculate accuracies
            # Create transform wrapper for GNN if needed
            transform_fn = None
            if args.use_gnn and GNN_AVAILABLE:
                def gnn_transform(x):
                    return x.squeeze(2).permute(0, 2, 1)
            else:
                gnn_transform = None
                
            results = {
                'epoch': global_step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None, transform_fn=gnn_transform),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None, transform_fn=gnn_transform),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None, transform_fn=gnn_transform),
                'total_cost_time': time.time() - step_start_time
            }
            
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
            global_step += 1  # Increment global step
    
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
                    # ======================= GNN INPUT TRANSFORMATION =======================
                    if args.use_gnn and GNN_AVAILABLE:
                        x = x.squeeze(2).permute(0, 2, 1)
                    # ========================================================================
                    features = algorithm.featurizer(x).detach().cpu().numpy()
                    source_features.append(features)
            source_features = np.concatenate(source_features, axis=0)
            
            # Extract features for target domain
            target_features = []
            with torch.no_grad():
                for data in target_loader:
                    x = data[0].cuda().float()
                    # ======================= GNN INPUT TRANSFORMATION =======================
                    if args.use_gnn and GNN_AVAILABLE:
                        x = x.squeeze(2).permute(0, 2, 1)
                    # ========================================================================
                    features = algorithm.featurizer(x).detach().cpu().numpy()
                    target_features.append(features)
            target_features = np.concatenate(target_features, axis=0)
            
            # Calculate h-divergence
            h_div, domain_acc = calculate_h_divergence(source_features, target_features)
            logs['h_divergence'].append(h_div)
            logs['domain_acc'].append(domain_acc)
            print(f"  H-Divergence: {h_div:.4f}, Domain Classifier Acc: {domain_acc:.4f}")
            
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
            transform_fn = None
            if args.use_gnn and GNN_AVAILABLE:
                def gnn_transform(x):
                    return x.squeeze(2).permute(0, 2, 1)
            else:
                gnn_transform = None
                
            # Compute SHAP values safely with optional transform
            shap_vals = safe_compute_shap_values(algorithm, background, X_eval, transform_fn=gnn_transform)
            
            # Convert to numpy safely before visualization
            X_eval_np = X_eval.detach().cpu().numpy()
            
            # Generate core visualizations
            plot_summary(shap_vals, X_eval_np, 
                         output_path=os.path.join(args.output, "shap_summary.png"))
            
            overlay_signal_with_shap(X_eval_np[0], shap_vals, 
                                    output_path=os.path.join(args.output, "shap_overlay.png"))
            
            plot_shap_heatmap(shap_vals, 
                             output_path=os.path.join(args.output, "shap_heatmap.png"))

            # Evaluate SHAP impact
            base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals, transform_fn=gnn_transform)
            
            # Save SHAP values
            save_path = os.path.join(args.output, "shap_values.npy")
            save_shap_numpy(shap_vals, save_path=save_path)
            
            # Compute impact metrics
            print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
            print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
            print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
            print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals, transform_fn=gnn_transform):.4f}")

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
                        x = x.squeeze(2).permute(0, 2, 1)
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
        args.use_gnn = False  # Default to CNN if not specified
        
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
    
    main(args)
