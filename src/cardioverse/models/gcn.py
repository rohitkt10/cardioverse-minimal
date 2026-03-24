import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dropout_edge


class GCNModel(nn.Module):
    """Standard Graph Convolutional Network for single-modality learning."""

    def __init__(self, config):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if config.bn else None

        # First layer
        self.convs.append(GCNConv(config.in_features, config.hidden_dim))
        if config.bn:
            self.bns.append(nn.BatchNorm1d(config.hidden_dim))

        # Hidden layers
        for _ in range(config.num_layers - 1):
            self.convs.append(GCNConv(config.hidden_dim, config.hidden_dim))
            if config.bn:
                self.bns.append(nn.BatchNorm1d(config.hidden_dim))

        self.dropout = config.dropout
        self.classifier = nn.Linear(config.hidden_dim, config.num_labels)

    def embedding(self, x, edge_index, batch=None, edge_dropout=0.0):
        """Return graph-level embedding."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if self.training and edge_dropout > 0:
            edge_index, _ = dropout_edge(edge_index, p=edge_dropout)

        # First layer
        x = self.convs[0](x, edge_index)
        if self.bns:
            x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Subsequent layers with residual connections
        for i, conv in enumerate(self.convs[1:], 1):
            x_res = x
            x = conv(x, edge_index)
            if self.bns:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res

        # Global pooling
        x = global_mean_pool(x, batch)
        return x

    def forward(self, x, edge_index, batch=None, edge_dropout=0.0):
        """Return logits."""
        z = self.embedding(x, edge_index, batch, edge_dropout)
        out = self.classifier(z)
        return out
