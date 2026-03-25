# Cardioverse-Minimal Architecture

This document defines the fixed architecture and usage patterns for cardioverse-minimal.

---

## Directory Structure

```
cardioverse-minimal/
├── data/
│   ├── gene_level_data/
│   │   ├── X.npy               # (n_samples, 916) Gene expression features
│   │   └── edge_index.npy      # (2, n_edges) Gene network edges
│   ├── reaction_level_data/
│   │   ├── X.npy               # (n_samples, 3572) Reaction features
│   │   └── edge_index.npy      # (2, n_edges) Reaction network edges
│   └── metadata/
│       └── drug_metadata.csv   # Drug name, cardiotoxicity label
│
├── src/cardioverse/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gcn.py              # GCNModel - standard graph conv
│   │   ├── linet.py            # LiNetModel - position-aware GNN with memory pooling
│   │   └── fusion.py           # ModalityFusionTransformer, GNNIntegrativeModel
│   ├── training/
│   │   ├── __init__.py
│   │   ├── gnn_trainer.py      # GNNTrainer - single-modality GNN training
│   │   └── fusion_trainer.py   # GNNFusionTrainer - two-stage multiview training
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── gnn_config.py       # GNNModelConfig, TrainingConfig
│   │   └── fusion_config.py    # FusionModelConfig, FusionTrainingConfig
│   ├── explanations/
│   │   ├── __init__.py
│   │   └── integrated_gradients.py  # IGExplainer
│   └── utils/
│       ├── __init__.py
│       └── regularization.py   # lp_regularizer
│
└── notebooks/
    ├── 01_single_modality_genes.ipynb
    ├── 02_single_modality_reactions.ipynb
    └── 03_multiview_fusion.ipynb
```

---

## Configs

### Single-Modality Configs (`configs/gnn_config.py`)

```python
@dataclass
class GNNModelConfig:
    """Config for single-modality GNN models (GCNModel, LiNetModel)."""
    in_features: int = 1
    hidden_dim: int = 100
    num_layers: int = 3
    dropout: float = 0.5
    bn: bool = True
    num_labels: int = 2

    # LiNet-specific (ignored by GCNModel)
    num_nodes: int = None
    pos_emb_size: int = 4
    num_clusters: int = 20
    pool_ratio: float = 0.5


@dataclass
class TrainingConfig:
    """Training config for single-modality GNN training."""
    batch_size: int = 64
    nepochs: int = 250
    lr: float = 5e-5
    lmbda_l1: float = 1e-3
    lmbda_l2: float = 1e-3
    lmbda_kl: float = 50.0
    logstep: int = 50
    edge_dropout: float = 0.0
```

### Multiview Configs (`configs/fusion_config.py`)

```python
@dataclass
class FusionModelConfig:
    """Config for fusion transformer (ModalityFusionTransformer)."""
    embed_dim: int = 100          # Must match GNNModelConfig.hidden_dim
    num_heads: int = 4
    dim_ff: int = 100
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    num_labels: int = 2


@dataclass
class FusionTrainingConfig:
    """Training config for two-stage multiview fusion training."""
    batch_size: int = 64
    nepochs_stage1: int = 100     # Freeze GNNs, train fusion only
    nepochs_stage2: int = 200     # Unfreeze all, train end-to-end
    lr_stage1: float = 5e-5
    lr_stage2: float = 1e-5
    lmbda_l1: float = 1e-3
    lmbda_l2: float = 1e-3
    lmbda_kl: float = 50.0
    logstep: int = 50
```

---

## Models

### GCNModel (`models/gcn.py`)

Standard Graph Convolutional Network for single-modality learning.

```python
class GCNModel(nn.Module):
    def __init__(self, config: GNNModelConfig):
        ...

    def embedding(self, x, edge_index, batch=None) -> Tensor:
        """Return graph-level embedding."""

    def forward(self, x, edge_index, batch=None) -> Tensor:
        """Return logits."""
```

**Usage:**
```python
from cardioverse.configs.gnn_config import GNNModelConfig
from cardioverse.models.gcn import GCNModel

config = GNNModelConfig(hidden_dim=100, num_layers=3)
model = GCNModel(config)
```

---

### LiNetModel (`models/linet.py`)

Position-aware GNN with memory pooling. Returns `(logits, kl_loss)` tuple.

```python
class LiNetModel(nn.Module):
    def __init__(self, config: GNNModelConfig):
        ...

    def embedding(self, x, edge_index, batch=None) -> Tuple[Tensor, Tensor]:
        """Return (graph_embedding, kl_loss)."""

    def forward(self, x, edge_index, batch=None) -> Tuple[Tensor, Tensor]:
        """Return (logits, kl_loss)."""
```

**Usage:**
```python
from cardioverse.configs.gnn_config import GNNModelConfig
from cardioverse.models.linet import LiNetModel

config = GNNModelConfig(
    hidden_dim=100,
    num_layers=3,
    num_nodes=916,  # Required for LiNet
    pos_emb_size=4,
    num_clusters=20,
    pool_ratio=0.5
)
model = LiNetModel(config)
```

---

### ModalityFusionTransformer (`models/fusion.py`)

Transformer with [CLS] token for fusing modality embeddings.

```python
class ModalityFusionTransformer(nn.Module):
    def __init__(self, config: FusionModelConfig):
        ...

    def embedding(self, x: Tensor) -> Tensor:
        """
        Compute fused embedding via transformer.

        Args:
            x: (B, M, D) where B=batch, M=num_modalities, D=hidden_dim

        Returns:
            (B, embed_dim) [CLS] token embedding
        """

    def forward(self, x: Tensor) -> Tensor:
        """Return logits."""
```

---

### GNNIntegrativeModel (`models/fusion.py`)

Wraps pretrained GNNs and fusion transformer for multiview learning.

```python
class GNNIntegrativeModel(nn.Module):
    def __init__(self, models: List[LiNetModel], model_config: FusionModelConfig):
        """
        Args:
            models: List of pretrained LiNetModel instances
            model_config: FusionModelConfig
        """
        ...

    def get_embeddings(self, graphs: List[Data]) -> Tuple[Tensor, List[Tensor]]:
        """
        Get embeddings from all GNNs.

        Returns:
            (stacked_embeddings, kl_losses)
            stacked_embeddings: (B, M, hidden_dim)
            kl_losses: List of per-modality KL losses
        """

    def forward(self, graphs: List[Data]) -> Tuple[Tensor, List[Tensor]]:
        """Return (logits, kl_losses)."""
```

**Usage:**
```python
from cardioverse.configs.fusion_config import FusionModelConfig
from cardioverse.models.fusion import GNNIntegrativeModel

fusion_config = FusionModelConfig(
    embed_dim=100,  # Must match pretrained GNN hidden_dim
    num_heads=4,
    dim_ff=100,
    num_layers=2
)

# models is a list of pretrained LiNetModel instances
fusion_model = GNNIntegrativeModel(models, fusion_config)
```

---

## Trainers

### GNNTrainer (`training/gnn_trainer.py`)

Single-modality GNN training. Handles both single-return (GCNModel) and tuple-return (LiNetModel) models.

```python
class GNNTrainer:
    def __init__(self, model: Union[GCNModel, LiNetModel], optimizer, edge_index: Tensor):
        ...

    def fit(self, train_dataset, val_dataset, config: TrainingConfig) -> Dict:
        """
        Train single-modality model.

        Returns:
            History dict with loss, acc, f1, auroc per epoch
        """

    def predict(self, x: Tensor) -> Tensor:
        """Return logits."""
```

**Usage:**
```python
from cardioverse.configs.gnn_config import GNNModelConfig, TrainingConfig
from cardioverse.models.linet import LiNetModel
from cardioverse.training.gnn_trainer import GNNTrainer
import torch.optim as optim

model_config = GNNModelConfig(hidden_dim=100, num_layers=3, num_nodes=916)
train_config = TrainingConfig(nepochs=250, lr=5e-5)

model = LiNetModel(model_config)
optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
trainer = GNNTrainer(model, optimizer, edge_index=torch.from_numpy(edge_index).long())

history = trainer.fit(train_dataset, val_dataset, train_config)
```

---

### GNNFusionTrainer (`training/fusion_trainer.py`)

Two-stage multiview training. Extends GNNTrainer with fusion-specific logic.

```python
class GNNFusionTrainer(GNNTrainer):
    def __init__(self, model: GNNIntegrativeModel, optimizer, edge_indices: List[Tensor]):
        """
        Args:
            model: GNNIntegrativeModel with pretrained GNNs
            edge_indices: List of edge_index tensors for each modality
        """
        ...

    def freeze_feature_extractors(self):
        """Freeze all GNN parameters, only train fusion."""

    def unfreeze_all(self):
        """Unfreeze all parameters."""

    def reset_optimizer(self, lr: float):
        """Recreate optimizer with new learning rate."""

    def fit(self, train_dataset, val_dataset, config: FusionTrainingConfig) -> Dict:
        """
        Two-stage training:
        - Stage 1: Freeze GNNs, train fusion only (nepochs_stage1, lr_stage1)
        - Stage 2: Unfreeze all, train end-to-end (nepochs_stage2, lr_stage2)

        Returns:
            {0: stage1_history, 1: stage2_history}
        """

    def predict(self, *xs: Tuple[Tensor, ...]) -> Tensor:
        """
        Args:
            *xs: Variable number of input tensors (one per modality)

        Returns:
            logits
        """
```

**Usage:**
```python
from cardioverse.configs.fusion_config import FusionModelConfig, FusionTrainingConfig
from cardioverse.models.fusion import GNNIntegrativeModel
from cardioverse.training.fusion_trainer import GNNFusionTrainer
import torch.optim as optim

fusion_config = FusionModelConfig(embed_dim=100, num_heads=4)
fusion_train_config = FusionTrainingConfig(
    nepochs_stage1=100,
    nepochs_stage2=200,
    lr_stage1=5e-5,
    lr_stage2=1e-5
)

# models is list of pretrained LiNetModel instances
fusion_model = GNNIntegrativeModel(models, fusion_config)
optimizer = optim.Adam(fusion_model.parameters(), lr=fusion_train_config.lr_stage1)

edge_indices = [torch.from_numpy(ei).long() for ei in [edge_index_genes, edge_index_reactions]]
trainer = GNNFusionTrainer(fusion_model, optimizer, edge_indices)

history = trainer.fit(train_dataset_mv, val_dataset_mv, fusion_train_config)
# history = {0: stage1_history, 1: stage2_history}
```

---

## Explanations

### IGExplainer (`explanations/integrated_gradients.py`)

Integrated Gradients attribution for GNN models.

```python
class IGExplainer:
    def __init__(self, model, edge_index: Tensor, baseline: float = 0.0):
        ...

    def explain(self, dataset, target: int = 1, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Compute integrated gradients.

        Returns:
            DataFrame with samples as rows, features as columns
        """
```

---

## Training Workflows

### Single-Modality Workflow (Genes OR Reactions)

```python
# 1. Load data
X = np.load("data/gene_level_data/X.npy")  # (n_samples, n_features)
edge_index = np.load("data/gene_level_data/edge_index.npy")
y = ...  # labels

# 2. Create datasets
train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).long(),
    torch.arange(len(X_train))
)
val_dataset = ...  # same structure

# 3. Configure and train
from cardioverse.configs.gnn_config import GNNModelConfig, TrainingConfig
from cardioverse.models.linet import LiNetModel
from cardioverse.training.gnn_trainer import GNNTrainer

model_config = GNNModelConfig(
    hidden_dim=100,
    num_layers=3,
    num_nodes=X.shape[1],
    num_clusters=20,
    pool_ratio=0.5
)
train_config = TrainingConfig(nepochs=250, lr=5e-5)

model = LiNetModel(model_config)
optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
trainer = GNNTrainer(model, optimizer, torch.from_numpy(edge_index).long())

history = trainer.fit(train_dataset, val_dataset, train_config)
```

---

### Multiview Workflow (Genes + Reactions)

**Step 1: Pretrain each modality individually**

```python
# Repeat single-modality workflow for each modality
models = {}
for name, X, edge_index in [("genes", X1, edge_index1), ("reactions", X2, edge_index2)]:
    model_config = GNNModelConfig(
        hidden_dim=100,  # Same for both!
        num_layers=3,
        num_nodes=X.shape[1],
        num_clusters=20,
        pool_ratio=0.5
    )
    train_config = TrainingConfig(nepochs=250, lr=5e-5)

    model = LiNetModel(model_config)
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
    trainer = GNNTrainer(model, optimizer, torch.from_numpy(edge_index).long())

    trainer.fit(train_dataset, val_dataset, train_config)
    models[name] = deepcopy(trainer.model).to("cpu")
```

**Step 2: Train fusion model**

```python
from cardioverse.configs.fusion_config import FusionModelConfig, FusionTrainingConfig
from cardioverse.models.fusion import GNNIntegrativeModel
from cardioverse.training.fusion_trainer import GNNFusionTrainer

# Create multiview dataset (both modalities)
train_dataset_mv = TensorDataset(
    torch.from_numpy(X1_train).float(),
    torch.from_numpy(X2_train).float(),
    torch.from_numpy(y_train).long(),
    torch.arange(len(X1_train))
)
val_dataset_mv = ...  # same structure

# Configure fusion
fusion_config = FusionModelConfig(
    embed_dim=100,  # Must match hidden_dim from pretraining!
    num_heads=4,
    dim_ff=100,
    num_layers=2
)
fusion_train_config = FusionTrainingConfig(
    nepochs_stage1=100,
    nepochs_stage2=200,
    lr_stage1=5e-5,
    lr_stage2=1e-5
)

# Create fusion model with pretrained GNNs
fusion_model = GNNIntegrativeModel(
    models=list(models.values()),
    model_config=fusion_config
)

# Train
optimizer = optim.Adam(fusion_model.parameters(), lr=fusion_train_config.lr_stage1)
edge_indices = [torch.from_numpy(ei).long() for ei in [edge_index1, edge_index2]]
fusion_trainer = GNNFusionTrainer(fusion_model, optimizer, edge_indices)

fusion_history = fusion_trainer.fit(train_dataset_mv, val_dataset_mv, fusion_train_config)
# Returns: {0: stage1_history, 1: stage2_history}
```

---

## Key Constraints

1. **`embed_dim` must equal `hidden_dim`**: When creating `FusionModelConfig`, set `embed_dim` to the same value as `GNNModelConfig.hidden_dim` used during pretraining.

2. **LiNetModel requires `num_nodes`**: This is the feature dimension (916 for genes, 3572 for reactions).

3. **GNNFusionTrainer expects LiNetModel instances**: The fusion model wraps `LiNetModel` instances specifically.

4. **Two-stage training is mandatory for multiview**: Stage 1 freezes GNNs, Stage 2 unfreezes all.

5. **Dataset format**:
   - Single-modality: `(X, y, sample_ids)`
   - Multiview: `(X1, X2, y, sample_ids)` - one tensor per modality

---

## Imports Reference

```python
# Single-modality
from cardioverse.configs.gnn_config import GNNModelConfig, TrainingConfig
from cardioverse.models.gcn import GCNModel
from cardioverse.models.linet import LiNetModel
from cardioverse.training.gnn_trainer import GNNTrainer

# Multiview
from cardioverse.configs.fusion_config import FusionModelConfig, FusionTrainingConfig
from cardioverse.models.fusion import ModalityFusionTransformer, GNNIntegrativeModel
from cardioverse.training.fusion_trainer import GNNFusionTrainer

# Explanations
from cardioverse.explanations.integrated_gradients import IGExplainer

# Utils
from cardioverse.utils.regularization import lp_regularizer
```
