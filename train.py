import time
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ, disable_inplace_relu
from datautil.getdataloader_single import get_act_dataloader
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Subset
from datautil.getcurriculumloader import get_curriculum_loader
from network.act_network import ActNetwork
from sklearn.metrics import davies_bouldin_score
import torchvision.transforms as transforms
from torchvision.models import resnet50
from collections import defaultdict
import torch.nn.functional as F

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

# ================ RESNET INTEGRATION ================
class ResNetModel(nn.Module):
    """ResNet integration with curriculum learning compatibility"""
    def __init__(self, num_classes, input_channels=12, use_pretrained=True):
        super(ResNetModel, self).__init__()
        # Load pre-trained ResNet-50
        self.resnet = resnet50(pretrained=use_pretrained)
        
        # Adjust first convolution for EMG channels
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize new conv layer
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Replace final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # Preprocessing transforms
        self.preprocess = nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        )
        
    def forward(self, x):
        # EMG data typically doesn't need the same preprocessing as images,
        # but we'll apply the transformation if the input has spatial dimensions
        if x.ndim == 4:  # [batch, channels, height, width]
            x = self.preprocess(x)
        return self.resnet(x)
    
    def predict(self, x):
        return self.forward(x)

# ================ CURRICULUM LEARNING COMPONENTS ================
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

def get_curriculum_loader(args, algorithm, train_dataset, val_dataset, stage):
    """
    High-performance curriculum learning system for 91%+ accuracy
    """
    # Group validation indices by domain
    domain_indices = defaultdict(list)
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        domain = item[2]  # Domain index
        domain_indices[domain].append(idx)

    domain_metrics = {}
    algorithm.eval()
    with torch.no_grad():
        for domain, indices in domain_indices.items():
            subset = torch.utils.data.Subset(val_dataset, indices)
            loader = DataLoader(subset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=args.N_WORKERS)

            total_loss = 0.0
            correct = 0
            total = 0
            num_batches = 0

            for batch in loader:
                inputs = batch[0].cuda().float()
                labels = batch[1].cuda().long()
                
                output = algorithm.predict(inputs)
                loss = torch.nn.functional.cross_entropy(output, labels)
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            accuracy = correct / total if total > 0 else 0
            domain_metrics[domain] = accuracy

    # Sort domains by accuracy (easiest first)
    sorted_domains = sorted(domain_metrics.keys(), 
                           key=lambda d: domain_metrics[d], 
                           reverse=True)
    
    print("\n--- Domain Ease Ranking (easiest to hardest) ---")
    for rank, domain in enumerate(sorted_domains, 1):
        print(f"{rank}. Domain {domain}: Accuracy = {domain_metrics[domain]:.4f}")

    # Conservative curriculum progression
    num_domains = len(sorted_domains)
    
    # Calculate domain inclusion based on training progress
    progress = min(1.0, stage / max(1, args.CL_PHASE_EPOCHS - 1))
    
    # Always include the easiest domain
    min_domains = 1
    
    # Gradually include more domains
    if stage < 3:  # First 3 stages: easiest domain only
        selected_domains = sorted_domains[:1]
    elif stage < 7:  # Next 4 stages: top 25% easiest domains
        selected_domains = sorted_domains[:max(min_domains, int(num_domains * 0.25))]
    elif stage < 12:  # Next 5 stages: top 50% easiest domains
        selected_domains = sorted_domains[:max(min_domains, int(num_domains * 0.5))]
    else:  # Full dataset
        selected_domains = sorted_domains
    
    print(f"Stage {stage}: Selecting {len(selected_domains)} domains")

    # Gather training indices from selected domains
    train_domain_indices = defaultdict(list)
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        domain = item[2]
        train_domain_indices[domain].append(idx)

    selected_indices = []
    for domain in selected_domains:
        if domain in train_domain_indices:
            # Use ALL samples from selected domains
            selected_indices.extend(train_domain_indices[domain])

    print(f"Using {len(selected_indices)} samples from {len(selected_domains)} domains")
    
    # Create curriculum subset
    curriculum_subset = SubsetWithLabelSetter(train_dataset, selected_indices)
    
    # Create a weighted sampler to balance classes
    class_counts = {}
    for idx in selected_indices:
        item = train_dataset[idx]
        label = item[1]
        class_counts[label] = class_counts.get(label, 0) + 1
        
    # Skip weighting if we have too few classes or samples
    if len(class_counts) < 2 or len(selected_indices) < 100:
        curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.N_WORKERS)
    else:
        weights = [1.0 / class_counts[train_dataset[idx][1]] for idx in selected_indices]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        curriculum_loader = DataLoader(curriculum_subset, batch_size=args.batch_size,
                                      sampler=sampler, num_workers=args.N_WORKERS)

    return curriculum_loader

# ================ MAIN TRAINING FUNCTION ================
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
        temp_model = ActNetwork(args.dataset).cuda()
        temp_model.eval()
        feature_list = []

        with torch.no_grad():
            for batch in train_loader:
                inputs = batch[0].cuda().float()
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

    # Initialize algorithm - use ResNet for high accuracy
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    
    # Determine number of classes
    num_classes = len(np.unique([item[1] for item in tr]))
    
    # Use ResNet if high accuracy mode is enabled
    if getattr(args, 'high_accuracy_mode', True):
        print("🚀 Using ResNet-50 for high accuracy training")
        # Create ResNet model separately
        resnet_model = ResNetModel(num_classes=num_classes, input_channels=tr[0][0].shape[0]).cuda()
        # Initialize algorithm normally
        algorithm = algorithm_class(args).cuda()
        # Replace the default network with ResNet
        algorithm.network = resnet_model
    else:
        print("Using standard ActNetwork")
        algorithm = algorithm_class(args).cuda()
        
    algorithm.train()

    # Setup optimizers
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    # Training metrics logging
    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'ent_loss', 
                           'total_loss', 'train_acc', 'valid_acc', 'target_acc', 
                           'total_cost_time']}
    best_valid_acc, target_acc = 0, 0

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
        for step in range(current_epochs):  # CHANGED: args.local_epoch -> current_epochs
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step, loss_result_dict['class']], colwidth=15)
            logs['class_loss'].append(loss_result_dict['class'])
    
        # Phase 2: Latent domain characterization
        print('==== Latent domain characterization ====')
        print_row(['epoch', 'total_loss', 'dis_loss', 'ent_loss'], colwidth=15)
        for step in range(current_epochs):  # CHANGED: args.local_epoch -> current_epochs
            for data in train_loader:
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
        for step in range(current_epochs):  # CHANGED: args.local_epoch -> current_epochs
            step_start_time = time.time()
            for data in train_loader:
                step_vals = algorithm.update(data, opt)
    
            # Calculate accuracies
            results = {
                'epoch': global_step,  # CHANGED: Use global step
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None),
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
                # Save best model
                torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))
            
            print_row([results[key] for key in print_key], colwidth=15)
            global_step += 1  # Increment global step
    
        logs['total_cost_time'].append(time.time() - round_start_time)
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
            
            # Compute SHAP values safely
            shap_vals = safe_compute_shap_values(algorithm, background, X_eval)
            
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
                x, y = data[0].cuda(), data[1]
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

    # Plot training metrics
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
    except Exception as e:
        print(f"[WARNING] Failed to generate training plots: {str(e)}")

    # Load best model for final evaluation
    algorithm.load_state_dict(torch.load(os.path.join(args.output, 'best_model.pth')))
    final_acc = modelopera.accuracy(algorithm, target_loader, None)
    print(f'\n🔥 Final Model Target Accuracy: {final_acc:.2%}')

if __name__ == '__main__':
    args = get_args()
    
    # Enable high accuracy mode by default
    if not hasattr(args, 'high_accuracy_mode'):
        args.high_accuracy_mode = True
        
    # Set curriculum learning parameters
    if not hasattr(args, 'curriculum'):
        args.curriculum = True
    if not hasattr(args, 'CL_PHASE_EPOCHS'):
        args.CL_PHASE_EPOCHS = 15  # Curriculum learning duration
        
    # Set epochs for high accuracy
    if args.high_accuracy_mode and not hasattr(args, 'max_epoch'):
        args.max_epoch = 100  # Longer training for ResNet
        
    main(args)
