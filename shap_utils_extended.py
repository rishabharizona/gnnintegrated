# Additional SHAP Evaluation Metrics

# ✅ Flip Rate (Prediction Instability)
def compute_flip_rate(base_preds, masked_preds):
    import numpy as np
    return np.mean(np.argmax(base_preds, axis=1) != np.argmax(masked_preds, axis=1))

# ✅ Prediction Confidence Change
def compute_confidence_change(base_preds, masked_preds):
    import numpy as np
    true_classes = np.argmax(base_preds, axis=1)
    conf_change = base_preds[np.arange(len(base_preds)), true_classes] - masked_preds[np.arange(len(base_preds)), true_classes]
    return np.mean(conf_change)

# ✅ AOPC (Area Over Perturbation Curve)
def compute_aopc(model, inputs, shap_values, evaluate_fn, max_k=20):
    import numpy as np
    shap_array = shap_values[0].values if isinstance(shap_values, list) else shap_values.values
    base_preds = model.predict(inputs).detach().cpu().numpy()
    base_acc = np.mean(np.argmax(base_preds, axis=1) == np.argmax(base_preds, axis=1))

    total_aopc = 0
    for k in range(1, max_k + 1):
        _, masked_preds, _ = evaluate_fn(model, inputs, shap_values, top_k=k)
        acc = np.mean(np.argmax(base_preds, axis=1) == np.argmax(masked_preds, axis=1))
        total_aopc += (base_acc - acc)
    return total_aopc / max_k

# ✅ Feature Coherence Score
def compute_feature_coherence(shap_array):
    import numpy as np
    diffs = np.abs(np.diff(shap_array.reshape(shap_array.shape[0], -1), axis=1))
    return np.mean(diffs)

# ✅ SHAP Entropy
def compute_shap_entropy(shap_array):
    import numpy as np
    norm = np.abs(shap_array) / np.sum(np.abs(shap_array), axis=1, keepdims=True)
    entropy = -np.sum(norm * np.log(norm + 1e-9), axis=1)
    return np.mean(entropy)
