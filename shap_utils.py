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
    """Improved handling for multi-class outputs"""
    if isinstance(shap_values, list):
        # Stack class explanations: (n_classes, samples, features) -> (samples, features, classes)
        return np.stack([v.values if hasattr(v, 'values') else v for v in shap_values], axis=-1)
    elif hasattr(shap_values, 'values'):
        return shap_values.values
    return shap_values

# ================= Visualization Functions =================

def plot_summary(shap_values, features, output_path, max_display=20):
    plt.figure(figsize=(10, 6))
    
    # Extract SHAP values array
    shap_array = _get_shap_array(shap_values)
    
    # Aggregate multi-class SHAP values
    if shap_array.ndim == 3:  # (samples, timesteps, classes)
        shap_array = np.abs(shap_array).max(axis=-1)  # Max importance across classes
    
    # Reshape features to match SHAP dimensions
    if features.ndim == 3:  # (samples, channels, timesteps)
        features = features.squeeze(1)  # Remove channel dimension
    features = features.reshape(features.shape[0], -1)
    
    # Reshape SHAP values
    flat_shap = shap_array.reshape(shap_array.shape[0], -1)
    
    # Create feature names
    feature_names = [f"T{t}" for t in range(features.shape[1])]
    
    # Create summary plot
    shap.summary_plot(
        flat_shap, 
        features,
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
    signal = to_numpy(signal)
    shap_vals = _get_shap_array(shap_vals)
    shap_vals = to_numpy(shap_vals)
    
    # Aggregate multi-class SHAP values if needed
    if shap_vals.ndim == 3:  # (samples, timesteps, classes)
        shap_vals = np.abs(shap_vals).max(axis=-1)  # Max importance across classes
    
    # Process first sample and channel
    if signal.ndim == 3:  # (samples, channels, timesteps)
        # For (200, 1, 1) shape
        if signal.shape[0] > signal.shape[2]:
            signal = signal[:, 0, 0]  # Time major format
        else:
            signal = signal[0, 0, :]  # Sample major format
    elif signal.ndim > 1:
        signal = signal.flatten()
    
    # Process SHAP values
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[0]  # First sample
    shap_vals = np.abs(shap_vals).flatten()
    
    # Ensure 1D arrays
    signal = signal.flatten()
    
    # Truncate to same length
    min_len = min(len(signal), len(shap_vals))
    signal = signal[:min_len]
    shap_vals = shap_vals[:min_len]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label="EMG Signal", color="steelblue", alpha=0.7, linewidth=1.5)
    plt.fill_between(
        np.arange(min_len), 
        0, 
        shap_vals, 
        color="red", 
        alpha=0.3, 
        label="|SHAP|"
    )
    plt.title("EMG Signal with SHAP Overlay (First Sample)")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved signal overlay: {output_path}")

def plot_shap_heatmap(shap_values, output_path):
    shap_vals = _get_shap_array(shap_values)
    abs_vals = np.abs(to_numpy(shap_vals))
    
    # Aggregate multi-class SHAP values
    if abs_vals.ndim == 3:  # (samples, timesteps, classes)
        abs_vals = abs_vals.max(axis=-1)  # Max importance per timestep
    
    # Average across samples
    aggregated = abs_vals.mean(axis=0)
    
    # Ensure 2D format (channels, timesteps)
    if aggregated.ndim == 1:
        aggregated = aggregated.reshape(1, -1)  # (1, timesteps)
    elif aggregated.ndim > 2:
        aggregated = aggregated.squeeze()  # Remove singleton dimensions
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(aggregated, cmap="viridis", cbar_kws={'label': '|SHAP Value|'})
    plt.xlabel("Time Steps")
    plt.ylabel("EMG Channels")
    plt.title("SHAP Value Heatmap (Average Across Samples)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved SHAP heatmap: {output_path}")

# ================== SHAP Impact Analysis ===================

def evaluate_shap_impact(model, inputs, shap_values, top_k=0.2):
    """Robust SHAP impact evaluation without device errors"""
    try:
        model.eval()
        device = next(model.parameters()).device
        
        # Get original predictions
        with torch.no_grad():
            base_preds = safe_model_predict(model, inputs)
            base_preds = torch.softmax(base_preds, dim=1)
        
        # Convert to numpy arrays
        base_preds_np = to_numpy(base_preds)
        inputs_np = to_numpy(inputs)
        shap_vals_np = to_numpy(shap_values)
        
        # Handle multi-class SHAP arrays
        if shap_vals_np.ndim > 4:
            shap_vals_np = np.abs(shap_vals_np).max(axis=1)
        elif shap_vals_np.ndim == 4 and shap_vals_np.shape[1] > 1:
            shap_vals_np = np.abs(shap_vals_np).max(axis=1)
        
        # Ensure proper dimensions
        if inputs_np.ndim < 2:
            inputs_np = inputs_np[np.newaxis, :]
        if shap_vals_np.ndim < 2:
            shap_vals_np = shap_vals_np[np.newaxis, :]
        
        # Get dimensions
        batch_size = inputs_np.shape[0]
        n_timesteps = inputs_np.shape[-1]
        
        # Reshape inputs to 4D: [batch, channels, spatial, time]
        inputs_4d = inputs_np.reshape(batch_size, -1, 1, n_timesteps)
        shap_4d = shap_vals_np.reshape(batch_size, -1, 1, n_timesteps)
        
        # Create masked inputs
        masked_inputs = inputs_4d.copy()
        
        # Process each sample
        for i in range(batch_size):
            importance = np.abs(shap_4d[i]).mean(axis=(0, 1))
            k = max(1, int(n_timesteps * top_k))
            top_indices = np.argsort(importance)[-k:]
            masked_inputs[i, :, :, top_indices] = 0
        
        # Convert back to tensor format - PyG safe method
        if hasattr(inputs, 'to_data_list'):  # PyG Batch object
            # Create new features tensor
            new_features = torch.tensor(
                masked_inputs.reshape(inputs.x.shape),
                dtype=inputs.x.dtype
            ).to(device)
            
            # Create new Batch with original structure
            masked_tensor = Batch(
                x=new_features,
                edge_index=inputs.edge_index,
                batch=inputs.batch,
                # Add other attributes as needed
            )
        else:  # Standard tensor
            original_shape = inputs.shape
            reshaped = masked_inputs.reshape(original_shape)
            masked_tensor = torch.tensor(reshaped, dtype=torch.float32).to(device)
        
        # Get predictions on masked inputs
        with torch.no_grad():
            masked_preds = safe_model_predict(model, masked_tensor)
            masked_preds = torch.softmax(masked_preds, dim=1)
        
        # Calculate metrics
        base_classes = base_preds.argmax(dim=1)
        masked_classes = masked_preds.argmax(dim=1)
        acc_drop = 100 * (1 - (base_classes == masked_classes).float().mean().item())
        
        return to_numpy(base_preds), to_numpy(masked_preds), acc_drop
        
    except Exception as e:
        print(f"SHAP impact evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, 0

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
    device = next(model.parameters()).device
    
    # Get original predictions
    with torch.no_grad():
        base_preds = model.predict(inputs)
        base_conf = torch.softmax(base_preds, dim=1).max(dim=1).values.cpu().numpy()
    
    # Convert SHAP to numpy and ensure contiguous memory
    shap_vals_np = to_numpy(shap_values)
    shap_vals_np = np.ascontiguousarray(shap_vals_np)
    
    if shap_vals_np.ndim == 3:
        shap_vals_np = np.abs(shap_vals_np).max(axis=-1)  # (samples, timesteps)
    
    aopc_scores = []
    
    # FIX: Ensure we only process available samples
    num_samples = min(len(base_conf), shap_vals_np.shape[0])
    
    for i in range(num_samples):  # Only process available samples
        # Get importance scores
        importance = shap_vals_np[i].flatten()
        sorted_indices = np.argsort(importance)[::-1]
        
        confidences = [base_conf[i]]
        
        # Create modified input
        if isinstance(inputs, (Data, Batch)):
            current = inputs.clone()
            # FIX: Handle batch indexing properly
            original_features = current.x.detach().clone().contiguous()
        else:
            current = inputs.clone()
            original_features = current.detach().clone().contiguous()
        
        for step in range(1, steps + 1):
            k = int(len(importance) * step / steps)
            mask_indices = sorted_indices[:k].copy()
            
            # Create modified features
            if isinstance(inputs, (Data, Batch)):
                modified_features = original_features.clone()
                # FIX: Correct indexing for PyG data
                if modified_features.dim() > 1:
                    modified_features[i, mask_indices] = 0
                else:
                    modified_features[mask_indices] = 0
                current.x = modified_features.contiguous()
            else:
                modified_features = original_features.clone()
                # FIX: Correct indexing for standard tensors
                if modified_features.dim() > 1:
                    modified_features[i, mask_indices] = 0
                else:
                    modified_features[mask_indices] = 0
                current = modified_features.contiguous()
            
            # Get prediction
            with torch.no_grad():
                pred = model.predict(current)
                conf = torch.softmax(pred, dim=1)[i].max().item()
            confidences.append(conf)
        
        # Calculate incremental drops
        incremental_drops = [confidences[j-1] - confidences[j] 
                           for j in range(1, len(confidences))]
        aopc = np.mean(incremental_drops) if incremental_drops else 0
        aopc_scores.append(aopc)
    
    return np.mean(aopc_scores) if aopc_scores else 0

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
    """4D interactive plot with robust dimension handling"""
    if not output_path.endswith('.html'):
        output_path += ".html"
    
    inputs = to_numpy(inputs)
    shap_vals = to_numpy(_get_shap_array(shap_values))
    
    print(f"[4D Plot] RAW Inputs shape: {inputs.shape}, SHAP shape: {shap_vals.shape}")
    
    # Handle multi-sample data by selecting first sample
    if inputs.ndim > 1 and inputs.shape[0] > 1:
        inputs = inputs[0]
    if shap_vals.ndim > 1 and shap_vals.shape[0] > 1:
        shap_vals = shap_vals[0]
    
    # Handle SHAP values - max across classes if needed
    if shap_vals.ndim > 1:
        shap_vals = np.abs(shap_vals).max(axis=0)  # Max across classes
    
    # Ensure proper dimensions
    if inputs.ndim == 0:
        inputs = np.array([inputs])
    
    # Create time steps array
    time_steps = np.arange(len(shap_vals))
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add EMG signal trace
    fig.add_trace(go.Scatter3d(
        x=time_steps,
        y=np.zeros_like(time_steps),
        z=inputs.flatten() if inputs.size > 1 else np.full_like(time_steps, inputs),
        mode='lines',
        name='EMG Signal',
        line=dict(width=2, color='blue'))
    )
    
    # Add SHAP values trace
    fig.add_trace(go.Scatter3d(
        x=time_steps,
        y=np.ones_like(time_steps),
        z=shap_vals,
        mode='lines+markers',
        name='SHAP Values',
        line=dict(width=4, color='red'))
    )
    
    fig.update_layout(
        title='4D EMG Signal and SHAP Values',
        scene=dict(
            xaxis_title='Time Steps',
            yaxis_title='Channel',
            zaxis_title='Value',
            yaxis=dict(tickvals=[0, 1], ticktext=['Signal', 'SHAP']),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        height=800,
        width=1000
    )
    
    fig.write_html(output_path)
    print(f"✅ Saved 4D SHAP plot: {output_path}")

def plot_4d_shap_surface(shap_values, output_path):
    """Surface plot with robust aggregation"""
    if not output_path.endswith('.html'):
        output_path += ".html"
    
    shap_vals = to_numpy(_get_shap_array(shap_values))
    
    print(f"[Surface] SHAP shape: {shap_vals.shape}")
    
    # Aggregate multi-class SHAP values
    if shap_vals.ndim == 3:
        shap_vals = np.abs(shap_vals).max(axis=-1)  # (samples, timesteps)
    
    # Aggregate across samples
    aggregated = np.abs(shap_vals).mean(axis=0)  # (timesteps,)
    
    # Create grid
    time_steps = np.arange(len(aggregated))
    channels = np.array([0, 1])  # Two channels for visualization
    
    X, Y = np.meshgrid(time_steps, channels)
    Z = np.array([aggregated, aggregated])  # (2, timesteps)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title='SHAP Value Surface',
        scene=dict(
            xaxis_title='Time Steps',
            yaxis_title='Channel',
            zaxis_title='|SHAP Value|',
            zaxis=dict(range=[0, aggregated.max() * 1.1])
        ),
        height=800,
        width=1000
    )
    
    fig.write_html(output_path)
    print(f"✅ Saved SHAP surface: {output_path}")

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
