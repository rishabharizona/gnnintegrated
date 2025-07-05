import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau, pearsonr, entropy as scipy_entropy
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import warnings
from scipy.stats import entropy
from torch_geometric.data import Data, Batch


def ensure_tensor_on_device(tensor, device):
    """Ensure tensor is on the specified device"""
    if isinstance(tensor, (Data, Batch)):
        return tensor.to(device)
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor

def safe_model_predict(model, inputs):
    """Safe prediction with device handling"""
    device = next(model.parameters()).device
    inputs = ensure_tensor_on_device(inputs, device)
    return model.predict(inputs)

# Helper function to safely convert tensors to numpy
def to_numpy(tensor):
    """Safely convert tensor to numpy array with detachment"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (Data, Batch)):
        # Handle PyG data types
        if hasattr(tensor, 'x'):
            return tensor.x.detach().cpu().numpy()
        elif hasattr(tensor, 'node_features'):
            return tensor.node_features.detach().cpu().numpy()
        return None
    return tensor

def extract_pyg_features(batch):
    """Extract feature tensor from PyG batch"""
    if hasattr(batch, 'x'):
        return batch.x
    elif hasattr(batch, 'node_features'):
        return batch.node_features
    # Try to find the feature tensor by name
    for attr in ['features', 'feat', 'data']:
        if hasattr(batch, attr):
            return getattr(batch, attr)
    return None

def get_pyg_batch_size(batch):
    """Get batch size from PyG data"""
    if hasattr(batch, 'batch'):
        return int(batch.batch.max()) + 1
    elif hasattr(batch, 'num_graphs'):
        return batch.num_graphs
    return 1

def safe_forward(model, x):
    """
    Forward pass that handles PyG data types
    """
    # For PyG data, use special handling
    if isinstance(x, (Data, Batch)) or hasattr(x, 'to_data_list'):
        return safe_forward_pyg(model, x)
    
    # Standard tensor handling
    x = x.clone().requires_grad_(True)
    
    # Disable inplace operations
    original_states = {}
    for name, module in model.named_modules():
        if hasattr(module, 'inplace'):
            original_states[name] = module.inplace
            module.inplace = False
    
    try:
        with torch.enable_grad():
            return model(x)
    finally:
        # Restore inplace states
        for name, module in model.named_modules():
            if name in original_states:
                module.inplace = original_states[name]

def safe_forward_pyg(model, data):
    """Special forward pass for PyG data"""
    # Disable inplace operations
    original_states = {}
    for name, module in model.named_modules():
        if hasattr(module, 'inplace'):
            original_states[name] = module.inplace
            module.inplace = False
    
    try:
        with torch.enable_grad():
            # Clone and set requires_grad on node features
            data = data.clone()
            if hasattr(data, 'x'):
                data.x = data.x.clone().requires_grad_(True)
            elif hasattr(data, 'node_features'):
                data.node_features = data.node_features.clone().requires_grad_(True)
            return model(data)
    finally:
        # Restore inplace states
        for name, module in model.named_modules():
            if name in original_states:
                module.inplace = original_states[name]

class PredictWrapper(torch.nn.Module):
    """Wrapper that handles PyG data for SHAP compatibility"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Handle PyG data differently
        if isinstance(x, (Data, Batch)) or hasattr(x, 'to_data_list'):
            return safe_forward_pyg(self.model, x)
        return safe_forward(self.model, x)

def get_background_batch(loader, size=64):
    """Get a batch of background samples for SHAP"""
    background = []
    for batch in loader:
        background.append(batch[0])
        if len(background) >= size:
            break
    
    # Handle PyG data
    if isinstance(background[0], (Data, Batch)) or hasattr(background[0], 'to_data_list'):
        return Batch.from_data_list(background[:size])
    return torch.cat(background, dim=0)[:size]

def safe_compute_shap_values(model, background, inputs):
    """
    Compute SHAP values safely with PyG support (final robust version)
    """
    try:
        # Get model device
        device = next(model.parameters()).device
        
        # Move data to model's device
        background = background.to(device)
        inputs = inputs.to(device)
        
        # Create prediction wrapper
        wrapped_model = PredictWrapper(model)
        
        # For PyG data, we need to convert to tensors for SHAP compatibility
        if isinstance(background, (Data, Batch)) or hasattr(background, 'to_data_list'):
            # Extract features from PyG data
            background_features = extract_pyg_features(background)
            inputs_features = extract_pyg_features(inputs)
            
            # Create a tensor-based wrapper that reconstructs PyG objects
            class TensorWrapper(torch.nn.Module):
                def __init__(self, model, background):
                    super().__init__()
                    self.model = model
                    self.background = background
                    self.orig_shape = extract_pyg_features(background).shape
                    self.device = next(model.parameters()).device
                 # Ensure background features are properly formatted
                if isinstance(background, (Data, Batch)):
                    background_features = extract_pyg_features(background)
                    if background_features.dim() == 2:
                        background_features = background_features.unsqueeze(0)  # Add batch dimension   
                        
                def forward(self, x):
                    # Convert numpy arrays to tensors
                    if isinstance(x, np.ndarray):
                        x = torch.tensor(x, dtype=torch.float32).to(self.device)

                    # Reshape to original graph structure
                    x = x.reshape(-1, *self.orig_shape[1:])  # Restore original dimensions
                    # Reconstruct PyG data from features
                    if isinstance(self.background, Batch):
                        # For batch data, reconstruct each graph
                        batch_list = []
                        start_idx = 0
                        for i in range(len(self.background)):
                            data = self.background[i].clone()
                            num_nodes = data.num_nodes
                            
                            # Handle different feature dimensions
                            if x.dim() == 3:
                                # Time series features: [batch, time, features]
                                node_features = x[start_idx:start_idx+num_nodes]
                            else:
                                # Standard features: [nodes, features]
                                node_features = x[start_idx:start_idx+num_nodes]
                            
                            start_idx += num_nodes
                            
                            if hasattr(data, 'x'):
                                data.x = node_features
                            elif hasattr(data, 'node_features'):
                                data.node_features = node_features
                            else:
                                # Try to set features by name
                                for attr in ['features', 'feat', 'data']:
                                    if hasattr(data, attr):
                                        setattr(data, attr, node_features)
                                        break
                            batch_list.append(data)
                        return self.model(Batch.from_data_list(batch_list))
                    else:
                        # For single graph
                        data = self.background.clone()
                        if hasattr(data, 'x'):
                            data.x = x
                        elif hasattr(data, 'node_features'):
                            data.node_features = x
                        else:
                            # Try to set features by name
                            for attr in ['features', 'feat', 'data']:
                                if hasattr(data, attr):
                                    setattr(data, attr, x)
                                    break
                        return self.model(data)
            
            # Create explainer with tensor-based wrapper
            tensor_wrapper = TensorWrapper(wrapped_model, background)
            
            # Create explainer - use KernelExplainer as final fallback
            try:
                # First try DeepExplainer
                explainer = shap.DeepExplainer(tensor_wrapper, background_features)
                shap_values = explainer.shap_values(inputs_features)
            except Exception as e:
                print(f"DeepExplainer failed: {str(e)}. Using KernelExplainer")
                # Convert to numpy for KernelExplainer
                bg_numpy = background_features.cpu().detach().numpy()
                inputs_numpy = inputs_features.cpu().detach().numpy()

                # FLATTEN TO 2D HERE
                bg_numpy = bg_numpy.reshape(bg_numpy.shape[0], -1)
                inputs_numpy = inputs_numpy.reshape(inputs_numpy.shape[0], -1)
                # Create model wrapper for KernelExplainer
                def model_wrapper(x):
                    return tensor_wrapper(x).detach().cpu().numpy()
                
                explainer = shap.KernelExplainer(
                    model_wrapper,
                    bg_numpy
                )
                shap_values = explainer.shap_values(inputs_numpy)
        else:
            # Standard tensor handling
            try:
                explainer = shap.DeepExplainer(wrapped_model, background)
                shap_values = explainer.shap_values(inputs)
            except Exception as e:
                print(f"DeepExplainer failed: {str(e)}. Using KernelExplainer")
                # Convert to numpy for KernelExplainer
                bg_numpy = background.cpu().detach().numpy()
                inputs_numpy = inputs.cpu().detach().numpy()
                
                # Create model wrapper for KernelExplainer
                def model_wrapper(x):
                    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                    return wrapped_model(x_tensor).detach().cpu().numpy()
                
                explainer = shap.KernelExplainer(
                    model_wrapper,
                    bg_numpy
                )
                shap_values = explainer.shap_values(inputs_numpy)
        
        return shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=to_numpy(inputs)
        )
    except Exception as e:
        print(f"SHAP computation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def _get_shap_array(shap_values):
    """Extract SHAP values array from Explanation object or list"""
    if isinstance(shap_values, list):
        return shap_values[0].values
    elif hasattr(shap_values, 'values'):
        return shap_values.values
    return shap_values

# ================= Visualization Functions =================

def plot_summary(shap_values, features, output_path, max_display=20):
    """Global feature importance summary plot (detached)"""
    plt.figure(figsize=(10, 6))
    
    # Extract SHAP values array
    shap_array = _get_shap_array(shap_values)
    
    # Reshape data for summary plot
    flat_features = features.reshape(features.shape[0], -1)
    flat_shap_values = shap_array.reshape(shap_array.shape[0], -1)
    
    # Verify shape consistency
    if flat_shap_values.shape != flat_features.shape:
        print(f"⚠️ Shape mismatch: SHAP values {flat_shap_values.shape} vs features {flat_features.shape}")
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
    
    # Create summary plot
    shap.summary_plot(
        flat_shap_values, 
        flat_features,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
        rng=42
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved summary plot: {output_path}")

def overlay_signal_with_shap(signal, shap_vals, output_path):
    """Overlay SHAP values on original signal (detached)"""
    signal = to_numpy(signal)
    shap_vals = _get_shap_array(shap_vals)
    shap_vals = to_numpy(shap_vals)
    
    # Handle different dimensions
    if signal.ndim > 1:
        signal = signal.squeeze()
    if shap_vals.ndim > 1:
        shap_vals = shap_vals.squeeze()
    
    # Flatten both arrays
    signal_flat = signal.reshape(-1)
    shap_vals_flat = np.abs(shap_vals).reshape(-1)  # Use absolute SHAP values
    
    # Truncate to same length
    min_len = min(len(signal_flat), len(shap_vals_flat))
    signal_flat = signal_flat[:min_len]
    shap_vals_flat = shap_vals_flat[:min_len]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot signal and SHAP overlay
    plt.plot(signal_flat, label="Signal", color="steelblue", alpha=0.7, linewidth=1.5)
    plt.fill_between(
        np.arange(min_len), 
        0, 
        shap_vals_flat, 
        color="red", 
        alpha=0.3, 
        label="|SHAP|"
    )
    
    plt.title("Signal with SHAP Overlay")
    plt.xlabel("Flattened Feature Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved signal overlay: {output_path}")

def plot_shap_heatmap(shap_values, output_path):
    """Heatmap of SHAP values across time and channels"""
    # Extract SHAP values array
    shap_vals = _get_shap_array(shap_values)
    
    # Convert to numpy and take absolute values
    abs_vals = np.abs(to_numpy(shap_vals))
    
    # Reduce to 2D: average across samples and spatial dimension
    while abs_vals.ndim > 2:
        abs_vals = abs_vals.mean(axis=tuple(range(abs_vals.ndim - 2)))
    
    # Now abs_vals should be 2D: (channels, time_steps)
    if abs_vals.ndim != 2:
        raise ValueError(f"Could not reduce SHAP values to 2D array. Final shape: {abs_vals.shape}")
    
    # Transpose to (channels, time_steps)
    aggregated = abs_vals.T
    
    plt.figure(figsize=(12, 8))
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
        base_preds = safe_model_predict(model, inputs)
        base_preds = torch.softmax(base_preds, dim=1)
    
    # Convert to numpy for processing
    base_preds_np = to_numpy(base_preds)
    inputs_np = to_numpy(inputs)
    shap_vals_np = to_numpy(shap_values)
    
    # Handle multi-class SHAP arrays
    if shap_vals_np.ndim > 4:
        # Reduce class dimension by taking max absolute values
        shap_vals_np = np.abs(shap_vals_np).max(axis=1)
    elif shap_vals_np.ndim == 4 and shap_vals_np.shape[1] > 1:
        # (batch, classes, ...) -> reduce to single importance
        shap_vals_np = np.abs(shap_vals_np).max(axis=1)
    
    # Ensure we have at least 2 dimensions
    if inputs_np.ndim < 2:
        inputs_np = inputs_np[np.newaxis, :]
    if shap_vals_np.ndim < 2:
        shap_vals_np = shap_vals_np[np.newaxis, :]
    
    # Get batch size
    batch_size = inputs_np.shape[0]
    
    # Reshape inputs to 4D: [batch, channels, spatial, time]
    # Without unpacking dimensions
    inputs_np = inputs_np.reshape(batch_size, -1, 1, inputs_np.shape[-1])
    
    # Reshape SHAP values to match inputs
    shap_vals_np = shap_vals_np.reshape(batch_size, -1, 1, inputs_np.shape[-1])
    
    # Now we can safely get dimensions
    n_channels = inputs_np.shape[1]
    n_timesteps = inputs_np.shape[3]
    
    # Rest of function remains unchanged...
    masked_inputs = inputs_np.copy()

     
    if inputs_np.shape != shap_vals_np.shape:
        # Find the smallest dimensions
        min_batch = min(inputs_np.shape[0], shap_vals_np.shape[0])
        min_channels = min(inputs_np.shape[1], shap_vals_np.shape[1])
        min_time = min(inputs_np.shape[3], shap_vals_np.shape[3])
        
        # Trim both arrays to match
        inputs_np = inputs_np[:min_batch, :min_channels, :, :min_time]
        shap_vals_np = shap_vals_np[:min_batch, :min_channels, :, :min_time]
        
        print(f"⚠️ Trimmed arrays to shape {inputs_np.shape}")
    # Mask top-K important features for each sample
    for i in range(batch_size):
        # Calculate importance per time step (average across channels and spatial)
        importance = np.abs(shap_vals_np[i]).mean(axis=(0, 1))
        
        # Ensure importance array matches time dimension
        if len(importance) > n_timesteps:
            importance = importance[:n_timesteps]
        
        # Determine threshold for top K%
        k = int(n_timesteps * top_k)
        top_indices = np.argsort(importance)[-k:]
        
        # Ensure indices are within valid range
        top_indices = top_indices[top_indices < n_timesteps]
        
        # Mask important timesteps across all channels
        masked_inputs[i, :, :, top_indices] = 0
    
    # Convert back to tensor
    # Handle PyG Data objects differently
    if isinstance(inputs, (Data, Batch)):
        # Create new PyG object with masked features
        masked_tensor = inputs.clone()
        if hasattr(masked_tensor, 'x'):
            masked_tensor.x = torch.tensor(masked_inputs, dtype=masked_tensor.x.dtype).to(masked_tensor.x.device)
        elif hasattr(masked_tensor, 'node_features'):
            masked_tensor.node_features = torch.tensor(masked_inputs, dtype=masked_tensor.node_features.dtype).to(masked_tensor.node_features.device)
    else:
        # Standard tensor handling
        masked_tensor = torch.tensor(masked_inputs, dtype=inputs.dtype).to(inputs.device)
    
    # Get predictions on masked inputs
    with torch.no_grad():
        masked_preds = safe_model_predict(model, masked_tensor)
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
    model.eval()
    
    # Handle 3D inputs by adding dummy spatial dimension
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(2)  # Add spatial dimension
    
    inputs_np = to_numpy(inputs)
    batch_size, n_channels, n_spatial, n_timesteps = inputs_np.shape
    device = inputs.device
    
    # Extract SHAP values array
    shap_vals_np = to_numpy(_get_shap_array(shap_values))
    
    # Handle 3D SHAP values
    if shap_vals_np.ndim == 3:
        shap_vals_np = shap_vals_np[:, :, np.newaxis, :]  # Match input dimensions
    
    with torch.no_grad():
        base_preds = model.predict(inputs)
        base_conf = torch.softmax(base_preds, dim=1).max(dim=1).values.cpu().numpy()
    
    aopc_scores = []
    
    for i in range(batch_size):
        # Get importance scores (average across channels and spatial)
        importance = np.abs(shap_vals_np[i]).mean(axis=(0, 1))
        
        # Ensure importance array matches time dimension
        if len(importance) > n_timesteps:
            importance = importance[:n_timesteps]
        
        sorted_indices = np.argsort(importance)[::-1].copy()
        mask_indices_tensor = torch.from_numpy(sorted_indices).to(device)
        
        current_input = inputs[i].clone().detach()
        original_conf = base_conf[i]
        confidences = [original_conf]
        
        # Gradually remove features
        for step in range(1, steps + 1):
            k = int(n_timesteps * step / steps)
            mask_indices = mask_indices_tensor[:k]
            
            # Ensure indices are within valid range
            mask_indices = mask_indices[mask_indices < n_timesteps]
            
            modified_input = current_input.clone()
            
            # Correct indexing based on tensor dimensions
            if modified_input.dim() == 3:  # (channels, spatial, timesteps)
                modified_input[:, :, mask_indices] = 0
            else:  # Handle unexpected dimensions
                modified_input[..., mask_indices] = 0
            
            # Get prediction
            with torch.no_grad():
                pred = model.predict(modified_input.unsqueeze(0))
                conf = torch.softmax(pred, dim=1).max().item()
            confidences.append(conf)
        
        # Calculate AOPC as average of incremental drops
        incremental_drops = []
        for j in range(1, len(confidences)):
            incremental_drop = confidences[j-1] - confidences[j]
            incremental_drops.append(incremental_drop)
        
        aopc = np.mean(incremental_drops) if incremental_drops else 0
        aopc_scores.append(aopc)
    
    return np.mean(aopc_scores)

# ================== Advanced Metrics ======================

def compute_shap_entropy(shap_values):
    """Compute entropy of SHAP value distribution"""
    # Extract SHAP values array
    abs_vals = np.abs(to_numpy(_get_shap_array(shap_values)))
    
    # Flatten spatial dimensions
    flat_vals = abs_vals.reshape(abs_vals.shape[0], -1)
    normalized = flat_vals / (flat_vals.sum(axis=1, keepdims=True) + 1e-10)
    ent = entropy(normalized, axis=1)
    return np.mean(ent)

def compute_feature_coherence(shap_values):
    """Measure spatial-temporal coherence of SHAP values"""
    # Extract SHAP values array
    vals = to_numpy(_get_shap_array(shap_values))
    
    # Compute channel-wise correlations
    channel_corrs = []
    for i in range(vals.shape[0]):
        # Remove singleton dimensions and ensure 2D shape
        chan_vals = vals[i].squeeze()  # Remove all singleton dimensions
        
        # If still more than 2D, flatten spatial dimensions
        if chan_vals.ndim > 2:
            chan_vals = chan_vals.reshape(chan_vals.shape[0], -1)
        
        # If we have only 1 channel, skip correlation calculation
        if chan_vals.shape[0] == 1:
            channel_corrs.append(1.0)  # Perfect correlation with itself
            continue
            
        # Compute pairwise channel correlations
        corr_matrix = np.corrcoef(chan_vals)
        np.fill_diagonal(corr_matrix, 0)
        channel_corrs.append(np.mean(corr_matrix))
    
    return np.mean(channel_corrs) if channel_corrs else 0

def compute_pca_alignment(shap_values):
    """Measure how well SHAP values align with PCA components"""
    # Extract SHAP values array
    vals = to_numpy(_get_shap_array(shap_values))
    
    # Flatten spatial dimensions
    flat_vals = vals.reshape(vals.shape[0], -1)
    
    # Compute PCA on absolute SHAP values
    pca = PCA(n_components=2)
    pca.fit(np.abs(flat_vals))
    
    # Compute explained variance ratio
    return pca.explained_variance_ratio_.sum()

def evaluate_advanced_shap_metrics(shap_values, inputs):
    """Compute a suite of advanced SHAP metrics and return Python floats"""
    # Extract SHAP values array
    shap_vals = to_numpy(_get_shap_array(shap_values))
    
    # Ensure inputs are in numpy format
    inputs_np = to_numpy(inputs)
    
    # Flatten inputs and SHAP values for mutual info
    flat_inputs = inputs_np.reshape(-1)
    flat_shap = np.abs(shap_vals).reshape(-1)
    
    # Limit to 1000 points to avoid memory issues
    if len(flat_inputs) > 1000:
        idx = np.random.choice(len(flat_inputs), 1000, replace=False)
        flat_inputs = flat_inputs[idx]
        flat_shap = flat_shap[idx]
    
    # Create bins for mutual information calculation
    input_min = np.min(flat_inputs)
    input_max = np.max(flat_inputs)
    input_bins = np.digitize(flat_inputs, bins=np.linspace(input_min, input_max, 10))
    
    shap_min = 0
    shap_max = np.max(flat_shap) + 1e-10  # Avoid division by zero
    shap_bins = np.digitize(flat_shap, bins=np.linspace(shap_min, shap_max, 10))
    
    # Compute all metrics
    metrics = {
        'shap_entropy': compute_shap_entropy(shap_values),
        'feature_coherence': compute_feature_coherence(shap_values),
        'channel_variance': np.var(shap_vals, axis=(0, 2, 3)).mean(),
        'temporal_entropy': entropy(np.abs(shap_vals).mean(axis=(0, 1, 2)).ravel()),
        'mutual_info': mutual_info_score(input_bins, shap_bins),
        'pca_alignment': compute_pca_alignment(shap_values)
    }
    
    # Convert all values to Python floats for safe formatting
    return {k: float(v) for k, v in metrics.items()}
# ================== 4D Visualizations =====================

def plot_emg_shap_4d(inputs, shap_values, output_path):
    """4D interactive plot of SHAP values using Plotly"""
    # Ensure HTML format
    if not output_path.endswith('.html'):
        output_path = os.path.splitext(output_path)[0] + ".html"
    
    inputs = to_numpy(inputs)
    
    # Extract SHAP values array
    shap_vals = to_numpy(_get_shap_array(shap_values))
    
    # For first sample only
    sample_idx = 0
    inputs = inputs[sample_idx]
    shap_vals = shap_vals[sample_idx]
    
    # Safely reduce dimensions - remove all singleton dimensions
    shap_vals = np.squeeze(shap_vals)
    
    # Get the number of time steps from input shape
    n_timesteps = inputs.shape[-1]
    
    # If SHAP values have more dimensions than expected, take first n_timesteps
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)
    elif shap_vals.ndim > 1:
        shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)
        shap_vals = shap_vals[:, :n_timesteps]
    
    # Ensure we have 2D array (channels, time_steps)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, n_timesteps)
    elif shap_vals.ndim > 2:
        shap_vals = shap_vals.reshape(-1, n_timesteps)
    
    n_channels = shap_vals.shape[0]
    
    # Create time steps array
    time_steps = np.arange(n_timesteps)
    
    # Create Plotly figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # Add traces for each channel
    for ch in range(n_channels):
        shap_mag = np.abs(shap_vals[ch])
        
        # Ensure arrays have same length
        if len(shap_mag) != len(time_steps):
            min_len = min(len(shap_mag), len(time_steps))
            shap_mag = shap_mag[:min_len]
            ch_time_steps = time_steps[:min_len]
        else:
            ch_time_steps = time_steps
        
        fig.add_trace(go.Scatter3d(
            x=ch_time_steps,
            y=np.full_like(ch_time_steps, ch),  # Constant channel index
            z=shap_mag,
            mode='lines',
            name=f'Channel {ch+1}',
            line=dict(width=4)
        ))
    
    # Set layout
    fig.update_layout(
        title='4D SHAP Value Distribution (Sample 0)',
        scene=dict(
            xaxis_title='Time Steps',
            yaxis_title='EMG Channels',
            zaxis_title='|SHAP Value|'
        ),
        height=800,
        width=1000
    )
    
    # Save as HTML
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"✅ Saved interactive 4D SHAP plot: {output_path}")

def plot_4d_shap_surface(shap_values, output_path):
    """Interactive surface plot of aggregated SHAP values using Plotly"""
    # Ensure HTML format
    if not output_path.endswith('.html'):
        output_path = os.path.splitext(output_path)[0] + ".html"
    
    # Extract SHAP values array
    shap_vals = to_numpy(_get_shap_array(shap_values))
    
    # Safely reduce dimensions
    shap_vals = np.squeeze(shap_vals)
    
    # Handle different dimensions
    if shap_vals.ndim == 4:
        # Average across spatial dimension: (batch, channels, spatial, time) -> (batch, channels, time)
        shap_vals = shap_vals.mean(axis=2)
    
    # Aggregate across samples
    if shap_vals.ndim == 3:
        # (batch, channels, time) -> (channels, time)
        aggregated = np.abs(shap_vals).mean(axis=0)
    elif shap_vals.ndim == 2:
        # (channels, time)
        aggregated = np.abs(shap_vals)
    else:
        raise ValueError(f"Unsupported SHAP dimension: {shap_vals.ndim}")
    
    # Ensure proper orientation: (channels, time)
    if aggregated.shape[0] > aggregated.shape[1]:
        aggregated = aggregated.T
    
    # Create grid
    channels = np.arange(aggregated.shape[0])
    time_steps = np.arange(aggregated.shape[1])
    X, Y = np.meshgrid(time_steps, channels)
    
    # Create Plotly surface plot
    fig = go.Figure(data=[
        go.Surface(
            z=aggregated,
            x=X,  # Time steps
            y=Y,  # Channels
            colorscale='Viridis',
            opacity=0.9,
            contours={
                "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
            }
        )
    ])
    
    # Customize layout
    fig.update_layout(
        title='SHAP Value Surface (Avg Across Samples)',
        scene=dict(
            xaxis_title='Time Steps',
            yaxis_title='EMG Channels',
            zaxis_title='|SHAP Value|',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.5)  # Adjust camera angle for better view
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800,
        width=1000
    )
    
    # Add colorbar
    fig.update_layout(coloraxis_colorbar=dict(
        title="|SHAP|",
        thickness=15,
        len=0.5
    ))
    
    # Save as HTML
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"✅ Saved interactive SHAP surface plot: {output_path}")

# ================== Similarity Metrics =====================

def compute_kendall_tau(shap1, shap2):
    """
    Compute Kendall's tau correlation between two SHAP arrays
    Args:
        shap1: First SHAP array (numpy array)
        shap2: Second SHAP array (numpy array)
    Returns:
        Kendall's tau correlation coefficient
    """
    # Flatten arrays and compute correlation
    flat1 = np.abs(shap1).flatten()
    flat2 = np.abs(shap2).flatten()
    return kendalltau(flat1, flat2)[0]

def cosine_similarity_shap(shap1, shap2):
    """
    Compute cosine similarity between two SHAP arrays
    Args:
        shap1: First SHAP array (numpy array)
        shap2: Second SHAP array (numpy array)
    Returns:
        Cosine similarity score (0-1)
    """
    # Flatten arrays and compute similarity
    flat1 = np.abs(shap1).flatten()
    flat2 = np.abs(shap2).flatten()
    return 1 - cosine(flat1, flat2)

def log_shap_values(shap_array):
    """
    Apply log transformation to SHAP values
    Args:
        shap_array: SHAP values array
    Returns:
        Log-transformed array (with safeguard for zero values)
    """
    # Take absolute value and add epsilon to avoid log(0)
    abs_shap = np.abs(shap_array)
    return np.log(abs_shap + 1e-12)

def compute_jaccard_topk(shap1, shap2, k=10):
    """
    Compute Jaccard similarity between top-k features of two SHAP arrays
    Args:
        shap1: First SHAP array (numpy array)
        shap2: Second SHAP array (numpy array)
        k: Number of top features to consider
    Returns:
        Jaccard similarity score (0-1)
    """
    # Flatten arrays and get top-k indices
    flat1 = np.abs(shap1).flatten()
    flat2 = np.abs(shap2).flatten()
    
    # Get top-k indices for each array
    top1 = set(np.argsort(-flat1)[:k])
    top2 = set(np.argsort(-flat2)[:k])
    
    # Compute Jaccard similarity
    intersection = len(top1.intersection(top2))
    union = len(top1.union(top2))
    return intersection / union if union > 0 else 0

# ✅ Save SHAP values
def save_shap_numpy(shap_values, save_path="shap_values.npy"):
    """Save SHAP values to numpy file"""
    shap_array = _get_shap_array(shap_values)
    np.save(save_path, shap_array)
    print(f"✅ Saved SHAP values to: {save_path}")
