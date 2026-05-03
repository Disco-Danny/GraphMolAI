from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import NNConv, global_add_pool, global_max_pool, global_mean_pool


class ResidualEdgeBlock(nn.Module):
    """Residual edge-conditioned message passing block."""

    def __init__(self, hidden_dim: int, num_edge_features: int, dropout: float) -> None:
        super().__init__()
        edge_network = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv = NNConv(hidden_dim, hidden_dim, edge_network, aggr="mean")
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        message = F.relu(self.conv(x, edge_index, edge_attr))
        message = F.dropout(message, p=self.dropout, training=self.training)
        x = self.message_norm(x + message)

        update = self.feed_forward(x)
        update = F.dropout(update, p=self.dropout, training=self.training)
        return self.feed_forward_norm(x + update)


class MPNNPlusPlus(nn.Module):
    """Stronger MPNN with residual blocks, normalization, and multi-pool readout."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualEdgeBlock(hidden_dim, num_edge_features, dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_encoder(x)
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        graph_embedding = torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch),
            ],
            dim=-1,
        )
        return self.readout(graph_embedding).view(-1)
