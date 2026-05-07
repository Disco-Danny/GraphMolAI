from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(nn.Module):
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
        hidden_dim = max(8, hidden_dim)
        heads = 4 if hidden_dim % 4 == 0 else 1
        out_dim = hidden_dim // heads
        self.dropout = dropout
        self.gat1 = GATConv(num_node_features, out_dim, heads=heads, dropout=dropout, edge_dim=num_edge_features)
        self.gat2 = GATConv(hidden_dim, out_dim, heads=heads, dropout=dropout, edge_dim=num_edge_features)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        return_attention: bool = False,
    ):
        if return_attention:
            x, attn1 = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, attn2 = self.gat2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x = F.elu(x)
        else:
            x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.gat2(x, edge_index, edge_attr=edge_attr))

        pooled = global_mean_pool(x, batch)
        out = self.head(pooled).view(-1)
        if return_attention:
            return out, [attn1, attn2]
        return out
