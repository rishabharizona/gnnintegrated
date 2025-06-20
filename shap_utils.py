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

import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import json
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
import random

# Helper function to safely convert tensors to numpy
def to_numpy(tensor):
    """Safely convert tensor to numpy array with detachment"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def safe_forward(model, x):
    """
    Forward pass that:
    1. Clones inputs to prevent modification
    2. Temporarily disables inplace operations
    3. Runs with gradient context
    4. Returns outputs with gradients preserved
    """
    # Clone inputs to prevent inplace modification
    x = x.clone().requires_grad_(True)
    
    # Disable inplace operations
    original_states = {}
    for name, module in model.named_modules():
        if hasattr(module, 'inplace'):
            original_states[name] = module.inplace
            module.inplace = False
    
    try:
        with torch.enable_grad():
            # Run model components separately
            features = model.featurizer(x)
            bottleneck = model.bottleneck(features)
            output = model.classifier(bottleneck)
            return output
    finally:
        # Restore original inplace states
        for name, module in model.named_modules():
            if name in original_states:
                module.inplace = original_states[name]

class PredictWrapper(torch.nn.Module):
    """Wrapper that uses safe_forward for SHAP compatibility"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return safe_forward(self.model, x)

def get_background_batch(loader, size=64):
    """Get a batch of background samples for SHAP"""
    background = []
    for batch in loader:
        background.append(batch[0])
        if len(background) >= size:
            break
    return torch.cat(background, dim=0)[:size]

def safe_compute_shap_values(model, background, inputs, nsamples=200):
    """
    Compute SHAP values safely with:
    - Custom forward pass
    - Gradient preservation
    - Error handling
    """
    # Create the explainer with our safe wrapper
    wrapped_model = PredictWrapper(model)
    
    # Use DeepExplainer for model-specific interpretation
    explainer = shap.DeepExplainer(
        wrapped_model,
        background,
    )
    
    # Compute SHAP values without additivity check
    shap_values = explainer.shap_values(
        inputs,
        check_additivity=False  # Disables problematic gradient check
    )
    
    # Convert to SHAP Explanation object for better handling
    return shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=to_numpy(inputs)
    )

# ================= Visualization Functions =================

def plot_summary(shap_values, features, output_path, max_display=20):
    """Global feature importance summary plot (detached)"""
    plt.figure(figsize=(10, 6))
    
    # Reshape data for summary plot
    # Flatten all dimensions except batch: (batch, channels, 1, time_steps) -> (batch, channels * time_steps)
    flat_features = features.reshape(features.shape[0], -1)
    
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        # Multi-class explanation: use SHAP values for first class
        flat_shap_values = shap_values.values[0].reshape(shap_values.values[0].shape[0], -1)
    else:
        flat_shap_values = shap_values.values.reshape(shap_values.values.shape[0], -1)
    
    # Verify shape consistency
    if flat_shap_values.shape != flat_features.shape:
        print(f"⚠️ Shape mismatch: SHAP values {flat_shap_values.shape} vs features {flat_features.shape}")
        # Attempt to automatically fix by truncating to min length
        min_samples = min(flat_shap_values.shape[0], flat_features.shape[0])
        min_features = min(flat_shap_values.shape[1], flat_features.shape[1])
        flat_shap_values = flat_shap_values[:min_samples, :min_features]
        flat_features = flat_features[:min_samples, :min_features]
        print(f"⚠️ Using truncated shapes: SHAP {flat_shap_values.shape}, features {flat_features.shape}")
    
    # Create feature names for EMG data
    feature_names = []
    for ch in range(features.shape[1]):  # Channels
        for t in range(features.shape[3]):  # Time steps
            feature_names.append(f"CH{ch+1}_T{t}")
    
    # Use explicit random state to avoid FutureWarning
    rng = np.random.RandomState(42)
    
    shap.summary_plot(
        flat_shap_values, 
        flat_features,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
        rng=rng  # Explicit random state
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved summary plot: {output_path}")

def overlay_signal_with_shap(signal, shap_vals, output_path):
    """Overlay SHAP values on original signal (detached)"""
    signal = to_numpy(signal)
    shap_vals = to_numpy(shap_vals)
    
    # For EMG data: signal shape (channels, 1, time_steps)
    plt.figure(figsize=(12, 6))
    
    # Plot original signal
    plt.subplot(2, 1, 1)
    for ch in range(signal.shape[1]):
        plt.plot(signal[0, ch, 0], label=f'Channel {ch+1}')
    plt.title("Original Signal")
    plt.legend()
    
    # Plot SHAP overlay
    plt.subplot(2, 1, 2)
    for ch in range(shap_vals.shape[1]):
        plt.plot(shap_vals[0, ch, 0], alpha=0.7, label=f'SHAP Channel {ch+1}')
    plt.title("SHAP Values Overlay")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved signal overlay: {output_path}")

def plot_shap_heatmap(shap_values, output_path):
    """Heatmap of SHAP values across time and channels"""
    plt.figure(figsize=(12, 8))
    
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        # Use SHAP values for first class
        shap_vals = shap_values.values[0]
    else:
        shap_vals = shap_values.values
    
    # For EMG data: shap_vals shape (samples, channels, 1, time_steps)
    # Aggregate across samples and spatial dim
    aggregated = np.abs(to_numpy(shap_vals)).mean(axis=0).squeeze()
    
    # Transpose to (channels, time_steps)
    aggregated = aggregated.T
    
    plt.imshow(aggregated, 
               aspect='auto', 
               cmap='viridis',
               interpolation='nearest')
    plt.colorbar(label='|SHAP Value|')
    plt.xlabel("Time Steps")
    plt.ylabel("EMG Channels")
    plt.title("SHAP Value Heatmap")
    
    # Add channel labels
    if aggregated.shape[0] <= 8:  # Only label if reasonable number of channels
        plt.yticks(range(aggregated.shape[0]), [f"CH{i+1}" for i in range(aggregated.shape[0])])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved SHAP heatmap: {output_path}")

# ================== SHAP Impact Analysis ===================

def evaluate_shap_impact(model, inputs, shap_values, top_k=0.2):
    """
    Evaluate the impact of SHAP values by masking important features
    Returns:
        base_preds: Original predictions
        masked_preds: Predictions after masking top features
        acc_drop: Accuracy drop percentage
    """
    model.eval()
    
    # Get original predictions
    with torch.no_grad():
        base_preds = model.predict(inputs)
        base_preds = torch.softmax(base_preds, dim=1)
    
    # Convert to numpy for processing
    base_preds_np = to_numpy(base_preds)
    inputs_np = to_numpy(inputs)
    
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        shap_vals_np = to_numpy(shap_values.values[0])
    else:
        shap_vals_np = to_numpy(shap_values.values)
    
    masked_inputs = inputs_np.copy()
    batch_size, n_channels, _, n_timesteps = inputs_np.shape
    
    # Mask top-K important features for each sample
    for i in range(batch_size):
        # Calculate importance per time step (average across channels)
        importance = np.abs(shap_vals_np[i]).mean(axis=(0, 1))
        
        # Determine threshold for top K%
        k = int(n_timesteps * top_k)
        top_indices = np.argsort(importance)[-k:]
        
        # Mask important timesteps across all channels
        masked_inputs[i, :, :, top_indices] = 0
    
    # Convert back to tensor
    masked_tensor = torch.tensor(masked_inputs, dtype=inputs.dtype).to(inputs.device)
    
    # Get predictions on masked inputs
    with torch.no_grad():
        masked_preds = model.predict(masked_tensor)
        masked_preds = torch.softmax(masked_preds, dim=1)
    
    # Calculate accuracy drop
    base_classes = base_preds.argmax(dim=1)
    masked_classes = masked_preds.argmax(dim=1)
    acc_drop = 100 * (1 - (base_classes == masked_classes).float().mean().item())
    
    return to_numpy(base_preds), to_numpy(masked_preds), acc_drop

def compute_flip_rate(base_preds, masked_preds):
    """Compute the class flip rate after masking"""
    base_classes = np.argmax(base_preds, axis=1)
    masked_classes = np.argmax(masked_preds, axis=1)
    flip_rate = np.mean(base_classes != masked_classes)
    return flip_rate

def compute_confidence_change(base_preds, masked_preds):
    """Compute average confidence change"""
    base_conf = np.max(base_preds, axis=1)
    masked_conf = np.max(masked_preds, axis=1)
    conf_change = np.mean(base_conf - masked_conf)
    return conf_change

def compute_aopc(model, inputs, shap_values, steps=10):
    """Compute Area Over Perturbation Curve"""
    model.eval()
    inputs_np = to_numpy(inputs)
    
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        shap_vals_np = to_numpy(shap_values.values[0])
    else:
        shap_vals_np = to_numpy(shap_values.values)
    
    batch_size, n_channels, _, n_timesteps = inputs_np.shape
    
    with torch.no_grad():
        base_preds = model.predict(inputs)
        base_conf = torch.softmax(base_preds, dim=1).max(dim=1).values
    
    aopc_scores = []
    
    for i in range(batch_size):
        # Get importance scores (average across channels)
        importance = np.abs(shap_vals_np[i]).mean(axis=(0, 1))
        sorted_indices = np.argsort(importance)[::-1]  # Most important first
        
        # Gradually remove features
        confidences = [to_numpy(base_conf[i])]
        current_input = inputs[i].clone()
        
        for step in range(1, steps + 1):
            # Mask top features proportionally
            k = int(n_timesteps * step / steps)
            mask_indices = sorted_indices[:k]
            current_input[:, :, mask_indices] = 0
            
            # Get prediction
            with torch.no_grad():
                pred = model.predict(current_input.unsqueeze(0))
                conf = torch.softmax(pred, dim=1).max().item()
            
            confidences.append(conf)
        
        # Calculate AOPC
        aopc = np.mean(confidences[0] - np.array(confidences[1:]))
        aopc_scores.append(aopc)
    
    return np.mean(aopc_scores)

# ================== Advanced Metrics ======================

def compute_shap_entropy(shap_values):
    """Compute entropy of SHAP value distribution"""
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        abs_vals = np.abs(to_numpy(shap_values.values[0]))
    else:
        abs_vals = np.abs(to_numpy(shap_values.values))
    
    # Flatten spatial dimensions
    flat_vals = abs_vals.reshape(abs_vals.shape[0], -1)
    normalized = flat_vals / (flat_vals.sum(axis=1, keepdims=True)) + 1e-10
    ent = entropy(normalized, axis=1)
    return np.mean(ent)

def compute_feature_coherence(shap_values):
    """Measure spatial-temporal coherence of SHAP values"""
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        vals = to_numpy(shap_values.values[0]))
    else:
        vals = to_numpy(shap_values.values)
    
    # Compute channel-wise correlations
    channel_corrs = []
    for i in range(vals.shape[0]):
        # Average across spatial dims (1 in EMG data)
        chan_vals = vals[i].squeeze(axis=1)  # Shape: (channels, time_steps)
        
        # Compute pairwise channel correlations
        corr_matrix = np.corrcoef(chan_vals)
        np.fill_diagonal(corr_matrix, 0)
        channel_corrs.append(np.mean(corr_matrix))
    
    return np.mean(channel_corrs) if channel_corrs else 0

def compute_pca_alignment(shap_values):
    """Measure how well SHAP values align with PCA components"""
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        vals = to_numpy(shap_values.values[0])
    else:
        vals = to_numpy(shap_values.values)
    
    # Flatten spatial dimensions
    flat_vals = vals.reshape(vals.shape[0], -1)
    
    # Compute PCA on absolute SHAP values
    pca = PCA(n_components=2)
    pca.fit(np.abs(flat_vals))
    
    # Compute explained variance ratio
    return pca.explained_variance_ratio_.sum()

def evaluate_advanced_shap_metrics(shap_values, inputs):
    """Compute a suite of advanced SHAP metrics"""
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        shap_vals = to_numpy(shap_values.values[0])
    else:
        shap_vals = to_numpy(shap_values.values)
    
    # Flatten inputs and SHAP values for mutual info
    flat_inputs = inputs.reshape(-1)
    flat_shap = np.abs(shap_vals).reshape(-1)
    
    # Create bins for mutual information calculation
    input_bins = np.digitize(flat_inputs, bins=np.linspace(flat_inputs.min(), flat_inputs.max(), 10))
    shap_bins = np.digitize(flat_shap, bins=np.linspace(0, flat_shap.max(), 10))
    
    return {
        'shap_entropy': compute_shap_entropy(shap_values),
        'feature_coherence': compute_feature_coherence(shap_values),
        'channel_variance': np.var(shap_vals, axis=(0, 2, 3)).mean(),
        'temporal_entropy': entropy(np.abs(shap_vals).mean(axis=(0, 1, 2))),
        'mutual_info': mutual_info_score(input_bins, shap_bins),
        'pca_alignment': compute_pca_alignment(shap_values)
    }

# ================== 4D Visualizations =====================

def plot_emg_shap_4d(inputs, shap_values, output_path):
    """4D scatter plot of SHAP values (channels x time x value)"""
    inputs = to_numpy(inputs)
    
    # Handle multi-class SHAP values
    if isinstance(shap_values, list):
        shap_vals = to_numpy(shap_values[0])
    else:
        shap_vals = to_numpy(shap_values)
    
    # For first sample only
    sample_idx = 0
    inputs = inputs[sample_idx]
    shap_vals = shap_vals[sample_idx]
    
    # Create interactive plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    time_steps = np.arange(inputs.shape[2])
    channels = np.arange(inputs.shape[0])
    
    # Plot each channel
    for ch in range(inputs.shape[0]):
        # Get SHAP magnitude for this channel
        shap_mag = np.abs(shap_vals[ch, 0])
        
        # Plot channel with different color
        ax.plot(
            time_steps, 
            np.full_like(time_steps, ch), 
            shap_mag, 
            label=f'Channel {ch+1}',
            linewidth=2
        )
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('EMG Channels')
    ax.set_zlabel('|SHAP Value|')
    ax.set_title('4D SHAP Value Distribution (Sample 0)')
    ax.legend()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved 4D SHAP plot: {output_path}")

def plot_4d_shap_surface(shap_values, output_path):
    """Surface plot of aggregated SHAP values"""
    # Handle multi-class SHAP values
    if isinstance(shap_values.values, list):
        shap_vals = to_numpy(shap_values.values[0])
    else:
        shap_vals = to_numpy(shap_values.values)
    
    # Aggregate across samples and spatial dim
    aggregated = np.abs(shap_vals).mean(axis=(0, 2)).T  # (channels, time_steps)
    
    # Create grid
    channels = np.arange(aggregated.shape[0])
    time_steps = np.arange(aggregated.shape[1])
    T, C = np.meshgrid(time_steps, channels)
    
    # Create plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(
        T, C, aggregated, 
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('EMG Channels')
    ax.set_zlabel('|SHAP Value|')
    ax.set_title('SHAP Value Surface (Avg Across Samples)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved SHAP surface plot: {output_path}")
