from dataclasses import dataclass, asdict


@dataclass
class GNNModelConfig:
    """
    Config for single-modality GNN models (GCNModel, LiNetModel).

    Args:
        in_features: Number of input features per node. Default: 1.
        hidden_dim: Hidden dimension size. Default: 100.
        num_layers: Number of graph convolutional layers. Default: 3.
        dropout: Dropout probability. Default: 0.5.
        bn: Whether to use batch normalization. Default: True.
        num_labels: Number of output classes. Default: 2.
        num_nodes: Number of nodes in the graph (LiNetModel only). Default: None.
        pos_emb_size: Size of positional embedding (LiNetModel only). Default: 4.
        num_clusters: Number of clusters for memory pooling (LiNetModel only). Default: 20.
        pool_ratio: Pooling ratio for TopK pooling (LiNetModel only). Default: 0.5.
    """
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

    def update(self, **kwargs):
        """Override config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")
        return self

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingConfig:
    """
    Training config for single-modality GNN training.

    Args:
        batch_size: Number of samples per batch. Default: 64.
        nepochs: Number of training epochs. Default: 250.
        lr: Learning rate. Default: 5e-5.
        lmbda_l1: L1 regularization coefficient. Default: 1e-3.
        lmbda_l2: L2 regularization coefficient. Default: 1e-3.
        lmbda_kl: KL divergence regularization coefficient (LiNetModel only). Default: 50.0.
        logstep: Log training metrics every N epochs. Default: 50.
        edge_dropout: Edge dropout probability (GCNModel only). Default: 0.0.
        checkpoint_after: Start checkpointing best model after this epoch. Default: 20.
    """
    batch_size: int = 64
    nepochs: int = 250
    lr: float = 5e-5
    lmbda_l1: float = 1e-3
    lmbda_l2: float = 1e-3
    lmbda_kl: float = 50.0
    logstep: int = 50
    edge_dropout: float = 0.0
    checkpoint_after: int = 20

    def update(self, **kwargs):
        """Override config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")
        return self

    def to_dict(self):
        return asdict(self)
