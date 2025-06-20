import time
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader
from torch.utils.data import DataLoader, ConcatDataset
from network.act_network import ActNetwork

# Full SHAP imports
from shap_utils import (
    get_shap_explainer, compute_shap_values, _get_shap_array, plot_summary,
    plot_force, evaluate_shap_impact, plot_shap_heatmap, get_background_batch,
    compute_jaccard_topk, compute_kendall_tau, cosine_similarity_shap,
    log_shap_numpy, overlay_signal_with_shap, compute_flip_rate, compute_confidence_change, compute_aopc, compute_feature_coherence, compute_shap_entropy,
    plot_emg_shap_4d, compute_shap_channel_variance, compute_shap_temporal_entropy, compare_top_k_channels, compute_mutual_info, compute_pca_alignment, plot_4d_shap_surface, safe_compute_shap_values
)

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)
    print_environ()
    print(s)
    os.makedirs(args.output, exist_ok=True)

    loader_data = get_act_dataloader(args)
    train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata = loader_data[:7]

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
        best_k, _ = max(((k, silhouette_score(all_features, KMeans(n_clusters=k, n_init=10).fit(all_features).labels_)) for k in range(2, 11)), key=lambda x: x[1])
        args.latent_domain_num = best_k
        del temp_model

    args.batch_size = 32 * args.latent_domain_num if args.latent_domain_num < 6 else 16 * args.latent_domain_num

    train_loader = DataLoader(tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=True)
    train_loader_noshuffle = DataLoader(tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    valid_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    target_loader = DataLoader(targetdata, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()

    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'ent_loss', 'total_loss', 'train_acc', 'valid_acc', 'target_acc', 'total_cost_time']}
    best_valid_acc, target_acc = 0, 0

    for round_idx in range(args.max_epoch):
        print(f'\n======== ROUND {round_idx} ========')

        if getattr(args, 'curriculum', False) and round_idx < getattr(args, 'CL_PHASE_EPOCHS', 5):
            algorithm.eval()
            train_loader = DataLoader(ConcatDataset([tr, val]), batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=True)
            algorithm.train()

        print('==== Feature update ====')
        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            logs['class_loss'].append(loss_result_dict['class'])

        print('==== Latent domain characterization ====')
        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            for key in ['dis', 'ent', 'total']:
                logs[f"{key}_loss"].append(loss_result_dict[key])

        algorithm.set_dlabel(train_loader)

        print('==== Domain-invariant feature learning ====')
        for step in range(args.local_epoch):
            for data in train_loader:
                step_vals = algorithm.update(data, opt)
            results = {
                'epoch': round_idx * args.local_epoch + step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None),
                'total_cost_time': time.time() - time.time()
            }
            for key in alg_loss_dict(args):
                results[f"{key}_loss"] = step_vals[key]
                logs[f"{key}_loss"].append(step_vals[key])
            for metric in ['train_acc', 'valid_acc', 'target_acc']:
                logs[metric].append(results[metric])
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']

    print(f'\n🎯 Final Target Accuracy: {target_acc:.4f}')

    if getattr(args, 'enable_shap', False):
        print("\n📊 Running SHAP explainability...")
        try:
            background = next(iter(valid_loader))[0][:64].cuda().float()
            background.requires_grad_(True)
            X_eval = background[:10].clone().detach().requires_grad_(True)
            for param in algorithm.parameters():
                param.requires_grad = True
            shap_vals = safe_compute_shap_values(algorithm, background, X_eval)
            plot_summary(shap_vals, X_eval.cpu().numpy(), os.path.join(args.output, "shap_summary.png"))
            plot_force(None, shap_vals, X_eval.cpu().numpy(), os.path.join(args.output, "shap_force.html"))
            overlay_signal_with_shap(X_eval[0].cpu().numpy(), shap_vals.values[0], os.path.join(args.output, "shap_overlay.png"))
            plot_shap_heatmap(shap_vals, os.path.join(args.output, "shap_heatmap.png"))
            base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals)
            print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
            print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
            print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
            print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals):.4f}")
            metrics = evaluate_advanced_shap_metrics(shap_vals, X_eval)
            for k, v in metrics.items():
                print(f"[SHAP] {k.replace('_', ' ').title()}: {v:.4f}")
            plot_emg_shap_4d(X_eval, shap_vals.values, os.path.join(args.output, "shap_4d_scatter.html"))
            plot_4d_shap_surface(shap_vals, os.path.join(args.output, "shap_4d_surface.html"))
        except Exception as e:
            import traceback
            print(f"[ERROR] SHAP analysis failed: {str(e)}")
            traceback.print_exc()

if __name__ == '__main__':
    args = get_args()
    main(args)
