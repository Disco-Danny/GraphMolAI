from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import NNConv, global_mean_pool


class MPNN(nn.Module):
    """Message Passing Neural Network with edge-conditioned filters."""

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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        self.edge_networks = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.grus = nn.ModuleList()

        for _ in range(num_layers):
            edge_network = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.edge_networks.append(edge_network)
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_network, aggr="mean"))
            self.grus.append(nn.GRU(hidden_dim, hidden_dim))

        self.readout = nn.Sequential(
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
    ) -> torch.Tensor:
        x = F.relu(self.node_encoder(x))
        hidden_state = x.unsqueeze(0)

        for conv, gru in zip(self.convs, self.grus):
            message = F.relu(conv(x, edge_index, edge_attr))
            message = F.dropout(message, p=self.dropout, training=self.training)
            x, hidden_state = gru(message.unsqueeze(0), hidden_state)
            x = x.squeeze(0)

        graph_embedding = global_mean_pool(x, batch)
        return self.readout(graph_embedding).view(-1)
