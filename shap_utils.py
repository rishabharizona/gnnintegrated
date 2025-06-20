import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score
import os
import warnings

# Suppress SHAP warnings
warnings.filterwarnings("ignore", message="torch==2.3.0", category=UserWarning)

# ✅ SHAP-safe wrapper (UPDATED)
class PredictWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Ensure gradients are enabled
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Enable gradients for input
        x = x.clone().detach().requires_grad_(True)
        return self.model.explain(x)

# ✅ Explainer setup (UPDATED)
def get_shap_explainer(model, background_data):
    model.eval()
    # Ensure gradients are enabled
    for param in model.parameters():
        param.requires_grad = True
        
    wrapped = PredictWrapper(model)
    return shap.DeepExplainer(wrapped, background_data)

def compute_shap_values(explainer, inputs):
    # Ensure gradients are enabled
    inputs = inputs.clone().detach().requires_grad_(True)
    return explainer(inputs)

def _get_shap_array(shap_values):
    if isinstance(shap_values, list):
        return shap_values[0].values
    return shap_values.values

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

# ✅ Force plot (first instance) (UPDATED)
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

# ✅ Mask most influential inputs and evaluate impact (UPDATED)
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

# ✅ Background data for DeepExplainer (UPDATED)
def get_background_batch(loader, size=100):
    x_bg = []
    for batch in loader:
        x = batch[0]
        # Ensure gradients are enabled
        x = x.clone().detach().requires_grad_(True)
        x_bg.append(x)
        if len(torch.cat(x_bg)) >= size:
            break
    return torch.cat(x_bg)[:size]

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

# ✅ Enable gradients for SHAP (NEW)
def enable_shap_gradients(model):
    """Enable gradients for SHAP analysis"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    print("[SHAP] Enabled gradients for all model parameters")

# ✅ Safe SHAP computation (NEW)
def safe_compute_shap_values(model, background, inputs):
    """Compute SHAP values with proper gradient handling"""
    # Enable gradients
    enable_shap_gradients(model)
    
    # Create explainer
    explainer = get_shap_explainer(model, background)
    
    # Compute SHAP values
    with torch.enable_grad():
        shap_values = compute_shap_values(explainer, inputs)
    
    return shap_values
