import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau, pearsonr, entropy as scipy_entropy
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.decomposition import PCA
import os
import warnings

# Suppress SHAP warnings
warnings.filterwarnings("ignore", message="torch==2.3.0", category=UserWarning)

# ==============================
# 🔍 Core SHAP Functionality
# ==============================

# ✅ SHAP-safe wrapper
class PredictWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Directly run the model components with gradient tracking
        with torch.enable_grad():
            features = self.model.featurizer(x)
            bottleneck = self.model.bottleneck(features)
            return self.model.classifier(bottleneck)

# ✅ Explainer setup
def get_shap_explainer(model, background_data):
    # Create wrapper that bypasses custom methods
    wrapped = PredictWrapper(model)
    return shap.DeepExplainer(wrapped, background_data)

# ✅ Safe SHAP computation
def safe_compute_shap_values(model, background, inputs):
    """Compute SHAP values with proper gradient handling"""
    # Create explainer
    explainer = get_shap_explainer(model, background)
    
    # Compute SHAP values with gradient context
    with torch.enable_grad():
        return explainer(inputs)

# ✅ Background data for DeepExplainer
def get_background_batch(loader, size=100):
    x_bg = []
    for batch in loader:
        x = batch[0]
        x_bg.append(x)
        if len(torch.cat(x_bg)) >= size:
            break
    return torch.cat(x_bg)[:size]

# ==============================
# 📊 Visualization Functions
# ==============================

# ✅ Summary plot
def plot_summary(shap_values, inputs, output_path="shap_summary.png", log_to_wandb=False):
    shap_array = _get_shap_array(shap_values)
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    flat_shap_values = shap_array.reshape(shap_array.shape[0], -1)

    if flat_inputs.shape[1] != flat_shap_values.shape[1]:
        print(f"[WARN] Adjusting flat_inputs from {flat_inputs.shape[1]} to {flat_shap_values.shape[1]}")
        repeat_factor = flat_shap_values.shape[1] // flat_inputs.shape[1]
        flat_inputs = np.repeat(flat_inputs, repeat_factor, axis=1)

    plt.figure()
    shap.summary_plot(flat_shap_values, flat_inputs, show=False)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    if log_to_wandb:
        wandb.log({"SHAP Summary": wandb.Image(output_path)})

# ✅ Force plot (first instance)
def plot_force(explainer, shap_values, inputs, index=0, output_path="shap_force.html", log_to_wandb=False):
    shap_array = _get_shap_array(shap_values)
    shap_for_instance = shap_array[index].reshape(-1)

    # Get expected value from explainer
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        expected_value = ev[0] if isinstance(ev, (list, np.ndarray)) else ev
    else:
        expected_value = 0

    # Get base value from model prediction
    with torch.enable_grad():
        base_value = explainer.model(inputs[index:index+1]).mean().item()

    # Use whichever is available
    base_val = expected_value if expected_value != 0 else base_value
        
    force_html = shap.plots.force(base_val, shap_for_instance)
    shap.save_html(output_path, force_html)

    if log_to_wandb:
        wandb.log({"SHAP Force Plot": wandb.Html(open(output_path).read())})

# ✅ Overlay SHAP importance on signal
def overlay_signal_with_shap(signal, shap_val, output_path="shap_overlay.png", log_to_wandb=False):
    signal = signal.reshape(-1)
    shap_val = shap_val.reshape(-1)

    # 🔧 Truncate to same length
    min_len = min(len(signal), len(shap_val))
    signal = signal[:min_len]
    shap_val = shap_val[:min_len]

    # 📊 Plot
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label="Signal", color="steelblue", alpha=0.7)
    plt.fill_between(np.arange(min_len), 0, shap_val, color="red", alpha=0.3, label="SHAP")

    plt.title("Signal with SHAP Overlay")
    plt.xlabel("Flattened Feature Index")
    plt.ylabel("Signal / SHAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    if log_to_wandb:
        wandb.log({"SHAP Overlay": wandb.Image(output_path)})

# ✅ SHAP heatmap: channels × time
def plot_shap_heatmap(shap_values, output_path="shap_heatmap.png", log_to_wandb=False):
    # Handle both SHAP Explanation objects and raw NumPy arrays
    if hasattr(shap_values, "values"):
        shap_array = shap_values.values
    else:
        shap_array = shap_values  # Already a NumPy array

    # Safety: Convert to NumPy and ensure float32/float64 dtype
    shap_array = np.array(shap_array, dtype=np.float32)

    # Reduce across samples and aux dimension → (channels, time)
    while shap_array.ndim > 2:
        shap_array = shap_array.mean(axis=0)

    # Ensure 2D: Now shap_array should be (channels, time)
    if shap_array.ndim != 2:
        raise ValueError(f"[plot_shap_heatmap] Expected 2D after reduction, got {shap_array.shape}")

    plt.figure(figsize=(10, 6))
    sns.heatmap(shap_array, cmap="coolwarm", cbar_kws={'label': 'SHAP Value'})
    plt.title("SHAP Heatmap (Mean across Samples & Aux)")
    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved SHAP heatmap to: {output_path}")

    if log_to_wandb:
        wandb.log({"SHAP Heatmap": wandb.Image(output_path)})

# ✅ 4D SHAP + Signal Visualizer
def plot_emg_shap_4d(signal, shap_val, sample_id=0, output_path="shap_4d_scatter.html", title="4D EMG SHAP Visualization", log_to_wandb=False):
    # Ensure we are dealing with numpy arrays
    if torch.is_tensor(signal):
        signal = signal.detach().cpu().numpy()
    if torch.is_tensor(shap_val):
        shap_val = shap_val.detach().cpu().numpy()
    
    # Select sample
    signal = signal[sample_id]   # shape: (C, T, A) or (C, T)
    shap_val = shap_val[sample_id]  # shape: (C, T, A) or (C, T, ...)

    # If there's an auxiliary dimension, average over it
    if shap_val.ndim == 3:
        shap_val = shap_val.mean(axis=-1)  # now (C, T)
    elif shap_val.ndim > 3:
        # If more than 3D, average over the extra dimensions
        shap_val = shap_val.mean(axis=tuple(range(2, shap_val.ndim)))
    
    # If signal has more than 2 dimensions, average over auxiliary
    if signal.ndim == 3:
        signal = signal.mean(axis=-1)  # (C, T)
    elif signal.ndim > 3:
        signal = signal.mean(axis=tuple(range(2, signal.ndim)))
    
    # Now both should be 2D: (C, T)
    n_channels, n_time = signal.shape

    data = {
        "Time": [], "Channel": [], "Signal": [], "SHAP": []
    }

    for c in range(n_channels):
        for t in range(n_time):
            data["Time"].append(t)
            data["Channel"].append(f"C{c}")
            data["Signal"].append(signal[c, t])
            data["SHAP"].append(shap_val[c, t])

    fig = px.scatter_3d(
        data,
        x="Time", y="Channel", z="Signal",
        color="SHAP",
        title=title,
        labels={"SHAP": "SHAP Importance"},
        color_continuous_scale="Inferno"
    )
    fig.update_traces(marker=dict(size=3))
    fig.write_html(output_path)
    print(f"[INFO] Saved 4D SHAP scatter plot to: {output_path}")

    if log_to_wandb:
        wandb.log({title: wandb.Html(open(output_path).read())})

# ✅ 4D SHAP Surface Plot
def plot_4d_shap_surface(shap_values, sample_id=0, output_path="shap_4d_surface.html", title="4D SHAP Surface", log_to_wandb=False):
    shap_array = _get_shap_array(shap_values)
    sample = shap_array[sample_id]  # shape: (C, T, A) or (C, T)

    # If there's an auxiliary dimension, average over it
    if sample.ndim == 3:
        sample = sample.mean(axis=-1)  # now (C, T)
    elif sample.ndim > 3:
        # If more than 3D, average over the extra dimensions
        sample = sample.mean(axis=tuple(range(2, sample.ndim)))
    elif sample.ndim < 2:
        raise ValueError(f"SHAP sample must be at least 2D, got {sample.ndim}D")

    # Now sample should be 2D: (channels, time)
    n_channels, n_time = sample.shape

    x = np.arange(n_time)  # Time
    y = np.arange(n_channels)  # Channels
    X, Y = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(z=sample, x=X, y=Y, colorscale='RdBu', colorbar=dict(title='SHAP'))])
    fig.update_layout(
        title=title,
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20),
        scene=dict(
            xaxis_title='Time Steps',
            yaxis_title='Channels',
            zaxis_title='SHAP Value',
        )
    )
    fig.write_html(output_path)
    print(f"[INFO] Saved 4D SHAP surface plot to: {output_path}")

    if log_to_wandb:
        wandb.log({title: wandb.Html(open(output_path).read())})

# ==============================
# 📈 Evaluation Functions
# ==============================

# ✅ Mask most influential inputs and evaluate impact
def evaluate_shap_impact(model, inputs, shap_values, top_k=10):
    # Use safe explain method with gradients disabled
    with torch.no_grad():
        base_preds = model.explain(inputs).detach().cpu().numpy()
        shap_array = _get_shap_array(shap_values)

        flat_shap = np.abs(shap_array).reshape(shap_array.shape[0], -1)
        sorted_indices = np.argsort(-flat_shap, axis=1)[:, :top_k]

        masked_inputs = inputs.clone()
        total_features = masked_inputs[0].numel()

        for i, indices in enumerate(sorted_indices):
            # Safe clamping
            indices = np.clip(indices, 0, total_features - 1)
            flat = masked_inputs[i].view(-1)
            flat[indices] = 0
            masked_inputs[i] = flat.view_as(masked_inputs[i])

        # Use safe explain method
        masked_preds = model.explain(masked_inputs).detach().cpu().numpy()
        accuracy_drop = np.mean(np.argmax(base_preds, axis=1) != np.argmax(masked_preds, axis=1))
    return base_preds, masked_preds, accuracy_drop

# ✅ Flip Rate (Prediction Instability)
def compute_flip_rate(base_preds, masked_preds):
    return np.mean(np.argmax(base_preds, axis=1) != np.argmax(masked_preds, axis=1))

# ✅ Prediction Confidence Change
def compute_confidence_change(base_preds, masked_preds):
    true_classes = np.argmax(base_preds, axis=1)
    conf_change = base_preds[np.arange(len(base_preds)), true_classes] - masked_preds[np.arange(len(base_preds)), true_classes]
    return np.mean(conf_change)

# ✅ AOPC (Area Over Perturbation Curve)
def compute_aopc(model, inputs, shap_values, max_k=20):
    """Compute Area Over the Perturbation Curve."""
    shap_array = _get_shap_array(shap_values)
    base_preds = model.explain(inputs).detach().cpu().numpy()
    base_acc = accuracy_score(np.argmax(base_preds, axis=1), np.argmax(base_preds, axis=1))

    total_aopc = 0
    for k in range(1, max_k + 1):
        _, masked_preds, _ = evaluate_shap_impact(model, inputs, shap_values, top_k=k)
        acc = accuracy_score(np.argmax(base_preds, axis=1), np.argmax(masked_preds, axis=1))
        total_aopc += (base_acc - acc)
    return total_aopc / max_k

# ✅ Feature Coherence Score
def compute_feature_coherence(shap_array):
    """Compute mean absolute difference between adjacent features in the SHAP array."""
    shap_array = _get_shap_array(shap_array)
    # Flatten each sample's SHAP values to 1D
    flat_shap = shap_array.reshape(shap_array.shape[0], -1)
    diffs = np.abs(np.diff(flat_shap, axis=1))
    return np.mean(diffs)

# ✅ SHAP Entropy
def compute_shap_entropy(shap_array):
    """Compute normalized entropy of SHAP values per sample."""
    shap_array = _get_shap_array(shap_array)
    flat_shap = shap_array.reshape(shap_array.shape[0], -1)
    norm = np.abs(flat_shap) / np.sum(np.abs(flat_shap), axis=1, keepdims=True)
    entropy = -np.sum(norm * np.log(norm + 1e-9), axis=1)
    return np.mean(entropy)

# ✅ SHAP Channel Variance
def compute_shap_channel_variance(shap_array):
    """Returns single scalar mean variance across all channels and samples."""
    shap_array = _get_shap_array(shap_array)
    # We have shape: (N, C, T, ...) -> reduce to (N, C, T)
    # If there's an auxiliary dimension, average over it
    if shap_array.ndim > 3:
        shap_array = shap_array.mean(axis=tuple(range(3, shap_array.ndim)))
    # Now (N, C, T)
    return shap_array.var(axis=2).mean()  # variance over time, then average over samples and channels

# ✅ SHAP Temporal Entropy
def compute_shap_temporal_entropy(shap_array):
    """Entropy over time for each channel's SHAP distribution, per sample, then averaged."""
    shap_array = _get_shap_array(shap_array)
    # Reduce auxiliary dimensions
    if shap_array.ndim > 3:
        shap_array = shap_array.mean(axis=tuple(range(3, shap_array.ndim)))
    # Now shape: (N, C, T)
    n_samples, n_channels, n_time = shap_array.shape
    entropies = []
    for i in range(n_samples):
        for c in range(n_channels):
            # Flatten the time dimension for this channel and sample
            channel_vals = shap_array[i, c, :]
            # Compute histogram of values
            probs, _ = np.histogram(channel_vals, bins=100, density=True)
            probs = probs[probs > 0]
            if len(probs) > 1:
                entropies.append(scipy_entropy(probs))
    return np.mean(entropies) if entropies else 0

# ✅ Compare top-k channels
def compare_top_k_channels(shap1, shap2, k=5):
    """Compares top-k channels between two SHAP instances (for the same input)."""
    shap1 = _get_shap_array(shap1)
    shap2 = _get_shap_array(shap2)
    # For each sample, we get the mean absolute SHAP per channel (averaging over time and auxiliary)
    shap1_mean = np.abs(shap1).mean(axis=tuple(range(2, shap1.ndim)))
    shap2_mean = np.abs(shap2).mean(axis=tuple(range(2, shap2.ndim)))
    jaccards = []
    for i in range(shap1_mean.shape[0]):
        top1 = set(np.argsort(-shap1_mean[i])[:k])
        top2 = set(np.argsort(-shap2_mean[i])[:k])
        jaccard = len(top1 & top2) / len(top1 | top2) if len(top1 | top2) > 0 else 0
        jaccards.append(jaccard)
    return np.mean(jaccards)

# ✅ Mutual Information
def compute_mutual_info(signal, shap_array):
    """Estimates mutual information between signal & SHAP values."""
    signal = signal.detach().cpu().numpy() if torch.is_tensor(signal) else signal
    shap_array = _get_shap_array(shap_array)
    # Flatten both to 1D arrays
    signal_flat = signal.ravel().astype(float)
    shap_flat = shap_array.ravel().astype(float)
    
    # Create bins for discretization
    bins_signal = np.histogram_bin_edges(signal_flat, bins=20)
    bins_shap = np.histogram_bin_edges(shap_flat, bins=20)
    
    # Digitize
    signal_binned = np.digitize(signal_flat, bins_signal)
    shap_binned = np.digitize(shap_flat, bins_shap)
    
    return mutual_info_score(signal_binned, shap_binned)

# ✅ PCA Alignment
def compute_pca_alignment(shap_array):
    """
    Measures alignment of SHAP values with the first principal component (PC1)
    across channels and time for each sample. Returns average absolute Pearson correlation.
    """
    shap_array = _get_shap_array(shap_array)
    # Reduce auxiliary dimensions
    if shap_array.ndim > 3:
        shap_array = shap_array.mean(axis=tuple(range(3, shap_array.ndim)))
    # Now shape: (N, C, T)
    B, C, T = shap_array.shape
    pca_scores = []
    for b in range(B):
        # Reshape to (T, C) - time steps as samples, channels as features
        sample = shap_array[b].T  # (T, C)
        if sample.shape[0] < 2 or sample.shape[1] < 2:
            pca_scores.append(0.0)
            continue
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(sample).flatten()       # (T,)
        # Compute the mean SHAP value over channels at each time step
        shap_mean = np.mean(shap_array[b], axis=0)      # (T,)
        if len(pc1) == len(shap_mean):
            corr, _ = pearsonr(pc1, shap_mean)
            pca_scores.append(abs(corr))
        else:
            pca_scores.append(0.0)
    return np.mean(pca_scores)

# ==============================
# 🔧 Utility Functions
# ==============================

# ✅ Similarity metrics
def compute_jaccard_topk(shap1, shap2, k=10):
    top1 = set(np.argsort(-np.abs(shap1.flatten()))[:k])
    top2 = set(np.argsort(-np.abs(shap2.flatten()))[:k])
    return len(top1 & top2) / len(top1 | top2)

def compute_kendall_tau(shap1, shap2):
    return kendalltau(shap1.flatten(), shap2.flatten())[0]

def cosine_similarity_shap(shap1, shap2):
    return 1 - cosine(shap1.flatten(), shap2.flatten())

# ✅ Save SHAP values
def log_shap_numpy(shap_values, save_path="shap_values.npy"):
    shap_array = _get_shap_array(shap_values)
    np.save(save_path, shap_array)

# ✅ Batch Metric Evaluation
def evaluate_advanced_shap_metrics(shap_values, signal_array):
    """
    Compute a batch of advanced metrics for SHAP values.
    Args:
        shap_values: SHAP values (Explanation object or array)
        signal_array: Corresponding input signal array (numpy or tensor)
    Returns:
        Dictionary of metrics
    """
    shap_array = _get_shap_array(shap_values)
    metrics = {
        "channel_variance": compute_shap_channel_variance(shap_array),
        "temporal_entropy": compute_shap_temporal_entropy(shap_array),
        "mutual_info": compute_mutual_info(signal_array, shap_array),
        "pca_alignment": compute_pca_alignment(shap_array),
        "feature_coherence": compute_feature_coherence(shap_array),
        "shap_entropy": compute_shap_entropy(shap_array),
    }
    return metrics
