import time
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ, disable_inplace_relu
from datautil.getdataloader_single import get_act_dataloader
from torch.utils.data import DataLoader
from datautil.getcurriculumloader import get_curriculum_loader
from network.act_network import ActNetwork
from sklearn.metrics import davies_bouldin_score
from shap_utils import (
    get_background_batch, safe_compute_shap_values, plot_summary,
    overlay_signal_with_shap, plot_shap_heatmap,
    evaluate_shap_impact, compute_flip_rate, compute_jaccard_topk,
    compute_kendall_tau, cosine_similarity_shap, save_shap_numpy, 
    compute_confidence_change, _get_shap_array, 
    compute_aopc, compute_feature_coherence, compute_shap_entropy,
    plot_emg_shap_4d, plot_4d_shap_surface, evaluate_advanced_shap_metrics
)

class SubsetWithLabelSetter(torch.utils.data.Subset):
    def __init__(self, dataset, indices, domain_label=None):
        super().__init__(dataset, indices)
        self.domain_label = domain_label

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        if self.domain_label is not None:
            return (data[0], data[1], self.domain_label)
        return data

def automated_k_estimation(features, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features)
        labels = kmeans.labels_
        if len(np.unique(labels)) < 2:
            continue
        silhouette = silhouette_score(features, labels)
        dbi = davies_bouldin_score(features, labels)
        norm_silhouette = (silhouette + 1) / 2
        norm_dbi = 1 / (1 + dbi)
        combined_score = (norm_silhouette + norm_dbi) / 2
        if combined_score > best_score:
            best_k = k
            best_score = combined_score
        print(f"K={k}: Silhouette={silhouette:.4f}, DBI={dbi:.4f}, Combined={combined_score:.4f}")
    print(f"[INFO] Optimal K determined as {best_k} (Score: {best_score:.4f})")
    return best_k

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
        args.latent_domain_num = automated_k_estimation(all_features)
        del temp_model

    args.batch_size = 32 * args.latent_domain_num if args.latent_domain_num < 6 else 16 * args.latent_domain_num
    print(f"Adjusted batch size: {args.batch_size}")

    def rebuild_loaders():
        return (
            DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.N_WORKERS, drop_last=False),
            DataLoader(tr, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS, drop_last=False),
            DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS, drop_last=False),
            DataLoader(targetdata, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS, drop_last=False)
        )

    train_loader, train_loader_noshuffle, valid_loader, target_loader = rebuild_loaders()

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()

    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'ent_loss', 'total_loss', 'train_acc', 'valid_acc', 'target_acc', 'total_cost_time']}
    best_valid_acc, target_acc = 0, 0
    global_step = 0

    for round_idx in range(args.max_epoch):
        print(f'\n======== ROUND {round_idx} ========')
        if getattr(args, 'curriculum', False):
            if round_idx < getattr(args, 'CL_PHASE_EPOCHS', 5):
                print(f"[Curriculum] Stage {round_idx}")
                train_loader = get_curriculum_loader(args, algorithm, tr, val, stage=round_idx)
                train_loader_noshuffle = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)
        current_epochs = args.CL_PHASE_EPOCHS if (getattr(args, 'curriculum', False) and round_idx < getattr(args, 'CL_PHASE_EPOCHS', 5)) else args.local_epoch

        print('==== Feature update ====')
        print_row(['epoch', 'class_loss'], colwidth=15)
        for step in range(current_epochs):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step, loss_result_dict['class']], colwidth=15)
            logs['class_loss'].append(loss_result_dict['class'])

        print('==== Latent domain characterization ====')
        print_row(['epoch', 'total_loss', 'dis_loss', 'ent_loss'], colwidth=15)
        for step in range(current_epochs):
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step, loss_result_dict['total'], loss_result_dict['dis'], loss_result_dict['ent']], colwidth=15)
            logs['dis_loss'].append(loss_result_dict['dis'])
            logs['ent_loss'].append(loss_result_dict['ent'])
            logs['total_loss'].append(loss_result_dict['total'])
        algorithm.set_dlabel(train_loader)

        print('==== Domain-invariant feature learning ====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch'] + [f"{item}_loss" for item in loss_list] + [f"{item}_acc" for item in eval_dict] + ['total_cost_time']
        print_row(print_key, colwidth=15)

        round_start_time = time.time()
        for step in range(current_epochs):
            step_start_time = time.time()
            for data in train_loader:
                step_vals = algorithm.update(data, opt)
            results = {
                'epoch': global_step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None),
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
            print_row([results[key] for key in print_key], colwidth=15)
            global_step += 1

        logs['total_cost_time'].append(time.time() - round_start_time)

    print(f'\n🎯 Final Target Accuracy: {target_acc:.4f}')

if __name__ == '__main__':
    args = get_args()
    main(args)
