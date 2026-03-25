# cardioverse-minimal

Preliminary GNN models and interpretability methods explored as part of the CARDIOVERSE project — an ARPA-H-funded effort. 

The core model is **LiNet**, a position-aware graph neural network with variational memory pooling, trained on bulk transcriptomic and metabolic reaction network data. Single-modality models (genes or reactions independently) can be further combined into a multiview fusion model via a Transformer that fuses per-modality embeddings using a [CLS]-token pooling mechanism. Post-hoc interpretability is done with **Integrated Gradients** (Captum), followed by Mann-Whitney U testing across correctly-classified toxic vs. non-toxic samples to rank the most informative genes and reactions.

---

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/).

**1. Install uv** (if you don't have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone and initialize:**

```bash
git clone <repo-url>
cd cardioverse-minimal
uv init --python 3.12
uv sync
uv sync --all-extras
```

`--all-extras` installs optional dependencies (Jupyter) needed to run the notebooks.

**3. Place the data in `data/`**

Refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for expected structure of data placement. 

**4. Run the notebooks:**

```bash
uv run jupyter lab
```

Then open any notebook from the `notebooks/` directory.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_single_modality_reactions.ipynb` | Train LiNet on metabolic reaction network data, run Integrated Gradients, identify top reactions |
| `02_single_modality_genes.ipynb` | Train LiNet on gene expression network data, run Integrated Gradients, identify top genes |
| `03_multiview_fusion.ipynb` | Pretrain both modalities, fuse via Transformer, run Integrated Gradients across both modalities |

---

## Usage

### Single-modality

```python
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from copy import deepcopy

from cardioverse.configs.gnn_config import GNNModelConfig, TrainingConfig
from cardioverse.models.linet import LiNetModel
from cardioverse.training.gnn_trainer import GNNTrainer

# Load data
X = pd.read_csv("data/gene_level_data/X.csv", index_col=0).values  # (n_samples, 916)
edge_index = np.load("data/gene_level_data/edge_index.npy")
y = ...  # (n_samples,) int64 labels

# Create datasets — format: (X, y, sample_ids)
X_train_t = torch.from_numpy(X_train).float()
train_dataset = TensorDataset(X_train_t, torch.from_numpy(y_train).long(), torch.arange(len(X_train_t)))
val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long(), torch.arange(len(X_val)))

# Configure
model_config = GNNModelConfig(
    hidden_dim=100,
    num_layers=3,
    dropout=0.5,
    num_nodes=X_train.shape[1],  # required for LiNet
)
train_config = TrainingConfig(nepochs=250, lr=1e-5)

# Train
model = LiNetModel(model_config)
optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
trainer = GNNTrainer(model, optimizer, torch.from_numpy(edge_index).long())
history = trainer.fit(train_dataset, val_dataset, train_config)

# Best model is restored automatically after training
model = deepcopy(trainer.model).to("cpu")
```

---

### Multiview fusion

```python
from cardioverse.configs.fusion_config import FusionModelConfig, FusionTrainingConfig
from cardioverse.models.fusion import GNNIntegrativeModel
from cardioverse.training.fusion_trainer import GNNFusionTrainer

# Step 1: pretrain each modality independently (repeat single-modality block above)
# models = {0: pretrained_genes_model, 1: pretrained_reactions_model}

# Step 2: multiview dataset — format: (X1, X2, y, sample_ids)
train_dataset_mv = TensorDataset(
    torch.from_numpy(X1_train).float(),
    torch.from_numpy(X2_train).float(),
    torch.from_numpy(y_train).long(),
    torch.arange(len(X1_train))
)
val_dataset_mv = TensorDataset(
    torch.from_numpy(X1_val).float(),
    torch.from_numpy(X2_val).float(),
    torch.from_numpy(y_val).long(),
    torch.arange(len(X1_val))
)

# Step 3: configure and train fusion
fusion_config = FusionModelConfig(
    embed_dim=100,   # must match hidden_dim used during pretraining
    num_heads=4,
    dim_ff=100,
    num_layers=2,
)
fusion_train_config = FusionTrainingConfig(
    nepochs_stage1=100,   # freeze GNNs, train transformer only
    nepochs_stage2=200,   # unfreeze all, end-to-end fine-tuning
    lr_stage1=5e-5,
    lr_stage2=1e-5,
)

fusion_model = GNNIntegrativeModel(models=list(models.values()), model_config=fusion_config)
optimizer = optim.Adam(fusion_model.parameters(), lr=fusion_train_config.lr_stage1)
edge_indices = [torch.from_numpy(edge_index1).long(), torch.from_numpy(edge_index2).long()]
fusion_trainer = GNNFusionTrainer(fusion_model, optimizer, edge_indices)

# Returns {0: stage1_history, 1: stage2_history}
fusion_history = fusion_trainer.fit(train_dataset_mv, val_dataset_mv, fusion_train_config)
```

---

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details on models, configs, trainers, and data flow.
