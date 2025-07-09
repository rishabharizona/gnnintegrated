
# GNNIntegrated: EMG-Based Activity Recognition with GNNs, Curriculum Learning, SHAP & Auto-K

## 📌 Overview

**GNNIntegrated** is a modular deep learning framework for **Human Activity Recognition (HAR)** from **Electromyography (EMG)** data. It leverages **Graph Neural Networks (GNNs)**, along with innovative extensions such as:

- **Curriculum Learning**: Progressive sample learning for better generalization.
- **Automated K Estimation**: Dynamic determination of cluster numbers.
- **SHAP Explainability**: Model interpretation via feature attribution.
- **Graph-Based Learning**: Temporal and structural dependencies through GNNs.

Abstract: Time series data frequently encounter changing conditions, causing traditional classification models to struggle when distributions shift, a critical challenge known as Out-of-Distribution (OOD) generalization. Although existing solutions such as DIVERSIFY (Lu et al., 2023) effectively identify latent domain structures, they fall short in adaptability, interpretability, and scalability, limiting its practical applications. To overcome these issues, we propose extending the DIVERSIFY framework with four key enhancements: Curriculum Learning, enabling gradual exposure to increasingly challenging data; Graph Neural Networks, capturing complex inter-variable relationships; SHAP-based explainability, providing clear insights into model decisions; and Automated K Estimation, removing manual tuning for domain partitioning. By integrating these techniques, our framework achieves higher autonomy, robustness, and transparency. We evaluated our method on real-world time series datasets, including Electromyography (EMG) and Human Activity Recognition (UCI-HAR), demonstrating significant improvements in OOD accuracy and generalization capabilities. Few enhancements have enabled the model to adapt seamlessly to unseen domains while providing meaningful explanations, significantly enhancing usability and reliability.

![image](https://github.com/user-attachments/assets/a5da6a74-70e4-4232-9e5b-153e616d297b)

---

## 🧠 Key Features

- **Temporal GCNs** for sequence-aware modeling of EMG signals.
- **Curriculum Learning** pipeline for progressively harder samples.
- **Auto-K Clustering** using Silhouette, CH, DB scores to auto-determine `k`.
- **SHAP Integration** to explain model decisions post-training.
- **Domain Adaptation & Diversification** for robust cross-subject learning.

---

## 🔧 Core Pipelines

### 🏋️ Training Pipeline (`train.py`) 

    
    ┌───────────────────────────────┐
    │        train.py               │
    ├───────────────────────────────┤
    │                               │
    │ 1. Argument Parsing           │
    │    • dataset, model, flags    │
    │                               │
    │ 2. Dataset Preparation        │
    │    • get_act_dataloader()     │
    │                               │
    │ 3. Graph Construction         │
    │    • graph_builder.py         │
    │    • temporal edges           │
    │                               │
    │ 4. Curriculum Learning        │
    │    • reorder samples          │
    │                               │
    │ 5. Model Initialization       │
    │    • ActNetwork (GNN)         │
    │    • domain adaptation opt.   │
    │                               │
    │ 6. Optimization               │
    │    • multi-component loss     │
    │      - alignment              │
    │      - classification         │
    │      - diversification        │
    │                               │
    │ 7. Logging                    │
    │    • loss, accuracy           │
    │    • clustering metrics       │
    │      - Silhouette             │
    │      - Calinski-Harabasz (CH) │
    │      - Davies-Bouldin (DB)    │
    └───────────────────────────────┘


### 📊 Evaluation Pipeline
    
    ┌───────────────────────────────┐
    │    evaluation pipeline        │
    ├───────────────────────────────┤
    │                               │
    │ 1. Load Best Model            │
    │    • resume checkpoint        │
    │                               │
    │ 2. Feature Extraction         │
    │    • GNN encoder embeddings   │
    │                               │
    │ 3. Auto-k Clustering          │
    │    • optimal cluster number   │
    │    • Silhouette, CH, DB       │
    │                               │
    │ 4. Classification             │
    │    • logistic regression      │
    │    • or NN classifier         │
    │                               │
    │ 5. SHAP Analysis              │
    │    • shap_utils.py            │
    │    • feature attributions     │
    │                               │
    │ 6. Visualization              │
    │    • confusion matrix         │
    │    • silhouette plots         │
    │    • SHAP explanations        │
    └───────────────────────────────┘


---

## 📁 File Structure

```
gnnintegrated-main/
├── train.py                  # Main pipeline
├── shap_utils.py            # SHAP explainability
├── env.yml                  # Environment setup
├── alg/                     # Domain adaptation, optimization
├── datautil/                # EMG dataset handling, clustering
├── gnn/                     # Graph construction, GNN models
├── loss/                    # Custom loss functions
├── network/                 # ActNetwork, adversarial nets
├── utils/                   # Arguments, reproducibility
└── README.md
```

---

## 📦 Dataset: EMG

The project is designed around **EMG (Electromyography) data**, processed into temporal sequences and graphs for HAR. Dataset loading is managed by:

```
datautil/actdata/
  ├── cross_people.py
  ├── util.py
```

Ensure your EMG data is structured or converted accordingly.

---

## 🧪 Extensions Breakdown

### 1. **GNN Backbone**

- Implements **Temporal GCNs** in `gnn/temporal_gcn.py`.
- Graphs constructed using `graph_builder.py`.

### 2. **Curriculum Learning**

- Sample difficulty progression implemented in `get_curriculum_loader()`.

### 3. **Automated K Estimation**

- Uses clustering metrics like:
  - **Silhouette Score**
  - **Calinski-Harabasz Index**
  - **Davies-Bouldin Score**
- Implemented in `train.py` and `datautil/cluster.py`.

### 4. **SHAP Integration**

- Post-hoc interpretability via `shap_utils.py`.
- Applies SHAP on feature embeddings for transparency.

---

## ▶️ How to Run

### 1. Install Environment

```bash
conda env create -f env.yml
conda activate base
```
### 2. Install Dependencies
```bash
# Clean up conflicting packages
pip uninstall -y thinc spacy accelerate peft fastai sentence-transformers

# Install core requirements
pip install numpy==1.26.3 --upgrade torch_geometric

pip install fastdtw scipy
```
### 3. Dataset download
EMG (electromyography)

     # Download the dataset
     wget https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip
     unzip diversity_emg.zip && mv emg data/
     
     # Create necessary directories
     mkdir -p ./data/train_output/act/
     
     mkdir -p ./data/emg
     mv emg/* ./data/emg
     
### 4. Train the Model

   Basic Execution
```bash
python train.py --data_dir ./data/ --task cross_people --test_envs 0 --dataset emg --algorithm diversify --alpha1 1.0 --alpha 1.0 --lam 0.0 --local_epoch 3 --max_epoch 3 --lr 0.0001 --output ./train_output --batch_size 64 --weight_decay 1e-3 --dropout 0.5 --label_smoothing 0.1 --automated_k --curriculum --CL_PHASE_EPOCHS 2 --enable_shap --use_gnn --gnn_hidden_dim 128 --gnn_output_dim 256 --gnn_pretrain_epochs 2

python train.py --data_dir ./data/ --task cross_people --test_envs 1 --dataset emg --algorithm diversify --alpha1 0.1 --alpha 10.0 --lam 0.0 --local_epoch 2 --max_epoch 5 --lr 0.01 --output ./data/train_output1 --batch_size 64 --weight_decay 1e-3 --dropout 0.5 --label_smoothing 0.1 --automated_k --curriculum --CL_PHASE_EPOCHS 3 --enable_shap --use_gnn --gnn_hidden_dim 128 --gnn_output_dim 256 --gnn_pretrain_epochs 3

python train.py --data_dir ./data/ --task cross_people --test_envs 2 --dataset emg --algorithm diversify --alpha1 0.5 --alpha 21.5 --lam 0.0 --local_epoch 4 --max_epoch 7 --lr 0.01 --output ./data/train_output2 --batch_size 64 --weight_decay 1e-3 --dropout 0.5 --label_smoothing 0.3 --automated_k --curriculum --CL_PHASE_EPOCHS 4 --enable_shap --use_gnn --gnn_hidden_dim 128 --gnn_output_dim 256 --gnn_pretrain_epochs 4

python train.py --data_dir ./data/ --task cross_people --test_envs 3 --dataset emg --algorithm diversify --alpha1 5.0 --alpha 0.1 --lam 0.0 --local_epoch 3 --max_epoch 9 --lr 0.01 --output ./data/train_output3 --batch_size 64 --weight_decay 1e-3 --dropout 0.5 --label_smoothing 0.1 --automated_k --curriculum --CL_PHASE_EPOCHS 5 --enable_shap --use_gnn --gnn_hidden_dim 128 --gnn_output_dim 256 --gnn_pretrain_epochs 5
```


Optional flags (see `utils/util.py::get_args()`):
- `--curriculum`
- `--enable_shap`
- `--gnn_hidden_dim`
- `--domain_adapt`

---

## 🧾 Outputs and Artifacts

- ✅ Trained model weights
- 📉 Loss and accuracy logs
- 📊 Clustering plots
- 📈 Confusion matrices
- 🔍 SHAP explanations
- 📁 Embedding files for downstream tasks

---

## 📈 Analysis & Visualization

- **Clustering Evaluation**:
  - Automatic `k` tuning
  - CH, DB, and Silhouette plotted
- **Classification Accuracy**:
  - Per-class metrics and confusion matrices
- **SHAP Analysis**:
  - Local/global feature importance
  - Interpretable visual outputs

---

## 🪪 License

This project is free for academic and commercial use with attribution.

            @misc{extdd2025,
              title={GNNIntegrated: EMG-Based Activity Recognition with GNNs, Curriculum Learning, SHAP & Auto-K},
              author={Rishabh Gupta et al.},
              year={2025},
              note={https://github.com/rishabharizona/gnnintegrated}
            }
