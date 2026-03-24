from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from cardioverse.models.linet import LiNetModel


class ModalityFusionTransformer(nn.Module):
    """Transformer with [CLS] token for fusing modality embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.cls_token = nn.Parameter(torch.randn(config.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.clf = nn.Linear(config.embed_dim, config.num_labels)

    def embedding(self, x: Tensor) -> Tensor:
        """
        Compute fused embedding via transformer.

        Args:
            x: (B, M, D) where B=batch, M=num_modalities, D=hidden_dim

        Returns:
            (B, embed_dim) [CLS] token embedding
        """
        B = x.size(0)
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, E)
        x = torch.cat((cls, x), dim=1)  # (B, M+1, E)
        x = self.encoder(x)
        return self.final_norm(x[:, 0])  # (B, E)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits."""
        z = self.embedding(x)
        return self.clf(z)


class GNNIntegrativeModel(nn.Module):
    """
    Wraps pretrained GNNs and fusion transformer for multiview learning.

    Args:
        models: List of pretrained LiNetModel instances
        model_config: FusionModelConfig
    """

    def __init__(self, models: List[LiNetModel], model_config):
        super().__init__()
        assert isinstance(models, list)
        models = [model.eval() for model in models]
        self.models = nn.ModuleList(models)
        self.transformer = ModalityFusionTransformer(model_config)

    def get_embeddings(self, graphs: List) -> Tuple[Tensor, List[Tensor]]:
        """
        Get embeddings from all GNNs.

        Args:
            graphs: List of torch_geometric.data.Batch objects (one per modality)

        Returns:
            (stacked_embeddings, kl_losses)
            stacked_embeddings: (B, M, hidden_dim)
            kl_losses: List of per-modality KL losses
        """
        embeddings = []
        kl_losses = []
        for graph, model in zip(graphs, self.models):
            args = (graph.x, graph.edge_index,)
            if hasattr(graph, "batch"):
                args = args + (graph.batch,)
            embedding, kl_loss = model.embedding(*args)
            embeddings.append(embedding)
            kl_losses.append(kl_loss)
        embeddings = torch.stack(embeddings, dim=1)
        return embeddings, kl_losses

    def forward(self, graphs: List) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass for multiview classification.

        Args:
            graphs: List of torch_geometric.data.Batch objects (one per modality)

        Returns:
            (logits, kl_losses)
        """
        embeddings, kl_losses = self.get_embeddings(graphs)
        logits = self.transformer(embeddings)
        return logits, kl_losses
