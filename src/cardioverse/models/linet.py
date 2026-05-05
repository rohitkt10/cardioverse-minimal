import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import numpy as np


class PosGCNConv(nn.Module):
    """Graph convolutional layer with position-guided community attention."""

    def __init__(self, in_channels, out_channels, pos_emb_size, normalize=True, bias=True):
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels, out_channels * pos_emb_size, bias=False)
        self.conv = gnn.GCNConv(out_channels, out_channels, bias=bias, normalize=normalize)

    def forward(self, x, edge_index, pos_embedding):
        pos_emb_size = pos_embedding.shape[-1]
        x = self.proj(x).view(-1, pos_emb_size, self.out_channels)
        x = x * pos_embedding.unsqueeze(-1)
        x = x.sum(dim=1)
        x = self.conv(x, edge_index)
        return x


class GNNBlock(nn.Module):
    """A single GNN block with PosGCNConv, BatchNorm, residual, and pooling."""

    def __init__(self, in_channels, out_channels, pos_emb_size, dropout=None,
                 batch_norm=True, residual=True, pool_ratio=0.5):
        super().__init__()
        self.conv = PosGCNConv(in_channels, out_channels, pos_emb_size)
        self.bn = nn.BatchNorm1d(out_channels) if batch_norm else None

        self.residual = residual
        if residual and in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = None

        self.pool = gnn.TopKPooling(out_channels, ratio=pool_ratio) if pool_ratio else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x, edge_index, batch, pos):
        z = x
        x = self.conv(x, edge_index=edge_index, pos_embedding=pos)
        if self.bn:
            x = self.bn(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        if self.residual:
            if self.res_proj:
                z = self.res_proj(z)
            x = x + z
        if self.pool:
            x, edge_index, _, batch, perm, _ = self.pool(x, edge_index=edge_index, batch=batch)
            pos = pos[perm]
        return x, edge_index, batch, pos


class LiNetModel(nn.Module):
    """Position-aware GNN with memory pooling."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert hasattr(config, "num_nodes"), "LiNet model requires fixed number of nodes"

        # Position embedding
        self.pos_embedding = nn.Sequential(
            nn.Embedding(config.num_nodes, config.pos_emb_size),
            nn.Softmax(dim=-1)
        )

        # Graph convolution blocks
        conv_blocks = []
        for i in range(config.num_layers):
            in_channels = 1 if i == 0 else config.hidden_dim
            conv_block = GNNBlock(
                in_channels=in_channels,
                out_channels=config.hidden_dim,
                pos_emb_size=config.pos_emb_size,
                dropout=0.3,
                batch_norm=True,
                residual=True,
                pool_ratio=config.pool_ratio,
            )
            conv_blocks.append(conv_block)
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # Memory pooling
        self.mempool = gnn.MemPooling(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim * 2,
            tau=1.0,
            heads=2,
            s=config.nnum_clusterum_clusters
        )
        self.fc1 = nn.Linear(self.mempool.out_channels * self.mempool.num_clusters, config.hidden_dim)
        self.global_residual = nn.Linear(config.num_nodes, config.hidden_dim)

        # Classifier
        self.clf = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.num_labels),
        )

    def embedding(self, x, edge_index, batch, **kwargs):
        """Return (graph_embedding, kl_loss)."""
        batch_size = len(np.unique(batch.cpu()))
        x_input = x.view(batch_size, -1)
        in_features = x.shape[0] // batch_size

        # Position embeddings
        pos = torch.vstack([torch.arange(in_features, device=x.device) for _ in range(batch_size)])
        pos = self.pos_embedding(pos)
        pos = pos.view(-1, pos.shape[-1])

        # Apply graph convolutions
        for conv_block in self.conv_blocks:
            x, edge_index, batch, pos = conv_block(x, edge_index, batch, pos)

        # Memory pooling
        x, score = self.mempool(x, batch)
        x = x.view(x.shape[0], -1)
        loss_kl = self.mempool.kl_loss(score)

        # Final embedding
        x = self.fc1(x)
        z = self.global_residual(x_input)
        x = x + z

        return x, loss_kl

    def forward(self, x, edge_index, batch, **kwargs):
        """Return (logits, kl_loss)."""
        x, loss_kl = self.embedding(x, edge_index, batch, **kwargs)
        x = self.clf(x)
        return x, loss_kl
