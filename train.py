import time
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader
from datautil.getcurriculumloader import get_curriculum_loader, split_dataset_by_domain
from torch.utils.data import ConcatDataset
from shap_utils import (
    get_shap_explainer, compute_shap_values, _get_shap_array, plot_summary,
    plot_force, evaluate_shap_impact, plot_shap_heatmap, get_background_batch,
    compute_jaccard_topk, compute_kendall_tau, cosine_similarity_shap,
    log_shap_numpy, overlay_signal_with_shap
)
from shap_utils_extended import (
    compute_flip_rate, compute_confidence_change, compute_aopc,
    compute_feature_coherence, compute_shap_entropy
)
from shap4D import (
    plot_emg_shap_4d, compute_shap_channel_variance, compute_shap_temporal_entropy,
    compare_top_k_channels, compute_mutual_info, compute_pca_alignment, plot_4d_shap_surface
)
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.io as pio
pio.renderers.default = 'colab'

def automated_k_estimation(features, k_min=2, k_max=10):
    """Automatically determine optimal cluster count using silhouette score"""
    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
        labels = kmeans.labels_
        score = silhouette_score(features, labels)

        if score > best_score:
            best_k = k
            best_score = score

    print(f"[INFO] Optimal K determined as {best_k} (Silhouette Score: {best_score:.4f})")
    return best_k

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)

    # Load datasets
    loader_data = get_act_dataloader(args)
    train_loader, train_loader_noshuffle, valid_loader, target_loader = loader_data[:4]
    tr, val, targetdata = loader_data[4:7] if len(loader_data) > 4 else (None, None, None)

    # Automated K estimation if enabled
    if getattr(args, 'automated_k', False):
        # Create a minimal feature extractor model
        from network.act_network import ActNetwork
        temp_model = ActNetwork(args.dataset).cuda()
        temp_model.eval()
        feature_list = []

        with torch.no_grad():
            for batch in train_loader:
                data = batch[0].cuda() if isinstance(batch, (list, tuple)) else batch.cuda()
                features = temp_model(data)
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
    for round_idx in range(args.max_epoch):
        print(f'\n======== ROUND {round_idx} ========')
        
        # Curriculum learning setup
        if getattr(args, 'curriculum', False) and round_idx < getattr(args, 'CL_PHASE_EPOCHS', 3):
            if tr is not None and val is not None:
                algorithm.eval()
                full_dataset = ConcatDataset([tr, val])
                tr_curriculum, val_curriculum = split_dataset_by_domain(full_dataset)
                train_loader = get_curriculum_loader(args, algorithm, tr_curriculum, val_curriculum, stage=round_idx)
                algorithm.train()
                print(f"Curriculum learning: Stage {round_idx}")

        # Phase 1: Feature update
        print('==== Feature update ====')
        print_row(['epoch', 'class_loss'], colwidth=15)
        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step, loss_result_dict['class']], colwidth=15)
            logs['class_loss'].append(loss_result_dict['class'])

        # Phase 2: Latent domain characterization
        print('==== Latent domain characterization ====')
        print_row(['epoch', 'total_loss', 'dis_loss', 'ent_loss'], colwidth=15)
        for step in range(args.local_epoch):
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
        for step in range(args.local_epoch):
            step_start_time = time.time()
            for data in train_loader:
                step_vals = algorithm.update(data, opt)

            results = {
                'epoch': round_idx * args.local_epoch + step,
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
            
            print_row([results[key] for key in print_key], colwidth=15)

        logs['total_cost_time'].append(time.time() - round_start_time)

    print(f'\n🎯 Final Target Accuracy: {target_acc:.4f}')

    # SHAP explainability analysis
    if getattr(args, 'enable_shap', False):
        print("\n📊 Running SHAP explainability...")
        try:
            background = get_background_batch(valid_loader, size=64).cuda()
            X_eval = background[:10]
            shap_explainer = get_shap_explainer(algorithm, background)
            shap_vals = compute_shap_values(shap_explainer, X_eval)
            shap_array = _get_shap_array(shap_vals)

            # Generate SHAP visualizations
            plot_summary(shap_vals, X_eval.cpu().numpy())
            plot_force(shap_explainer, shap_vals, X_eval.cpu().numpy())
            overlay_signal_with_shap(X_eval[0].cpu().numpy(), shap_array[0], 
                                    output_path="shap_overlay_sample0.png")

            # Evaluate SHAP impact
            base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals)
            log_shap_numpy(shap_vals)

            # Compute SHAP metrics
            print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
            print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
            print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
            print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals, evaluate_shap_impact):.4f}")
            print(f"[SHAP] Entropy: {compute_shap_entropy(shap_array):.4f}")
            print(f"[SHAP] Coherence: {compute_feature_coherence(shap_array):.4f}")

            # Multi-sample comparisons
            if len(shap_array) > 1:
                print(f"[SHAP] Jaccard: {compute_jaccard_topk(shap_array[0], shap_array[1]):.4f}")
                print(f"[SHAP] Kendall’s Tau: {compute_kendall_tau(shap_array[0], shap_array[1]):.4f}")
                print(f"[SHAP] Cosine Sim: {cosine_similarity_shap(shap_array[0], shap_array[1]):.4f}")

            # 4D-specific analysis
            try:
                plot_emg_shap_4d(X_eval, shap_array)
                plot_4d_shap_surface(shap_vals, output_path="shap_4d_surface.html")
                
                shap_array_reshaped = shap_array.reshape(shap_array.shape[0], -1, shap_array.shape[2])
                print(f"[SHAP4D] Channel Variance: {compute_shap_channel_variance(shap_array):.4f}")
                print(f"[SHAP4D] Temporal Entropy: {compute_shap_temporal_entropy(shap_array_reshaped):.4f}")
                
                signal_sample = X_eval[0].cpu().numpy()
                shap_sample = shap_array[0].mean(axis=-1)
                print(f"[SHAP4D] Mutual Info: {compute_mutual_info(signal_sample, shap_sample):.4f}")
                
                shap_array_reduced = shap_array.mean(axis=-1)
                print(f"[SHAP4D] PCA Alignment: {compute_pca_alignment(shap_array_reduced):.4f}")
            except Exception as e:
                print(f"[WARNING] 4D SHAP analysis failed: {str(e)}")

            # Confusion matrix
            true_labels, pred_labels = [], []
            for data in valid_loader:
                x, y = data[0].cuda(), data[1]
                preds = algorithm.predict(x).cpu()
                true_labels.extend(y.cpu().numpy())
                pred_labels.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())

            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix (Validation Set)")
            plt.savefig("confusion_matrix.png", dpi=300)
            plt.close()

        except Exception as e:
            print(f"[ERROR] SHAP analysis failed: {str(e)}")

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
        plt.savefig("training_metrics_plot.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"[WARNING] Failed to generate training plots: {str(e)}")

if __name__ == '__main__':
    args = get_args()
    
    # Add integrated features as command-line arguments
    if not hasattr(args, 'automated_k'):
        args.automated_k = False  # Default: manual K selection
    if not hasattr(args, 'curriculum'):
        args.curriculum = False   # Default: no curriculum learning
    if not hasattr(args, 'enable_shap'):
        args.enable_shap = False  # Default: disable SHAP
        
    main(args)
