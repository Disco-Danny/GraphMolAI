from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


class HybridGAT(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        fingerprint_dim: int,
        hidden_dim: int = 32,
        heads: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        out_channels = hidden_dim // heads
        self.dropout = dropout
        self.gat1 = GATConv(num_node_features, out_channels, heads=heads, dropout=dropout, edge_dim=num_edge_features)
        self.gat2 = GATConv(hidden_dim, out_channels, heads=heads, dropout=dropout, edge_dim=num_edge_features)
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.fp_proj = nn.Sequential(nn.Linear(fingerprint_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
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
        fingerprint: torch.Tensor,
        return_attention: bool = False,
    ):
        if return_attention:
            x, attn1 = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual = x
            x, attn2 = self.gat2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x = F.elu(x + residual)
        else:
            x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual = x
            x = F.elu(self.gat2(x, edge_index, edge_attr=edge_attr) + residual)
        graph_embedding = self.graph_norm(global_mean_pool(x, batch))
        fp_embedding = self.fp_proj(fingerprint)
        fused = torch.cat([fp_embedding, 0.3 * graph_embedding], dim=-1)
        out = self.head(fused).view(-1)
        if return_attention:
            return out, [attn1, attn2]
        return out
