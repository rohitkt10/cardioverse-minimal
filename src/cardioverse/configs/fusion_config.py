from dataclasses import dataclass, asdict


@dataclass
class FusionModelConfig:
    """
    Config for fusion transformer (ModalityFusionTransformer).

    Args:
        embed_dim: Embedding dimension. Must match GNNModelConfig.hidden_dim. Default: 100.
        num_heads: Number of attention heads. Default: 4.
        dim_ff: Feed-forward dimension. Default: 100.
        num_layers: Number of transformer layers. Default: 2.
        dropout: Dropout probability. Default: 0.1.
        activation: Activation function. Default: "gelu".
        num_labels: Number of output classes. Default: 2.
    """
    embed_dim: int = 100
    num_heads: int = 4
    dim_ff: int = 100
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    num_labels: int = 2

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
class FusionTrainingConfig:
    """
    Training config for two-stage multiview fusion training.

    Args:
        batch_size: Number of samples per batch. Default: 64.
        nepochs_stage1: Epochs for stage 1 (freeze GNNs, train fusion). Default: 100.
        nepochs_stage2: Epochs for stage 2 (unfreeze all, train end-to-end). Default: 200.
        lr_stage1: Learning rate for stage 1. Default: 5e-5.
        lr_stage2: Learning rate for stage 2. Default: 1e-5.
        lmbda_l1: L1 regularization coefficient. Default: 1e-3.
        lmbda_l2: L2 regularization coefficient. Default: 1e-3.
        lmbda_kl: KL divergence regularization coefficient. Default: 50.0.
        logstep: Log training metrics every N epochs. Default: 50.
    """
    batch_size: int = 64
    nepochs_stage1: int = 100
    nepochs_stage2: int = 200
    lr_stage1: float = 5e-5
    lr_stage2: float = 1e-5
    lmbda_l1: float = 1e-3
    lmbda_l2: float = 1e-3
    lmbda_kl: float = 50.0
    logstep: int = 50

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
