from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data


def _node_scores(num_nodes: int, attentions: list[tuple[torch.Tensor, torch.Tensor]]) -> list[float]:
    scores = torch.zeros(num_nodes, dtype=torch.float32)
    counts = torch.zeros(num_nodes, dtype=torch.float32)
    for edge_index, alpha in attentions:
        weights = alpha.mean(dim=-1).detach().cpu()
        edges = edge_index.detach().cpu()
        for idx in range(edges.size(1)):
            src = int(edges[0, idx])
            dst = int(edges[1, idx])
            value = float(weights[idx])
            scores[src] += value
            scores[dst] += value
            counts[src] += 1.0
            counts[dst] += 1.0
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    scores = scores / counts
    if torch.max(scores) > 0:
        scores = scores / torch.max(scores)
    return scores.tolist()


def visualize_attention(model, data: Data, output_path: str | Path, device: torch.device) -> tuple[list[float], Path]:
    model.eval()
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        _, attentions = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
            batch,
            data.fingerprint.to(device).view(1, -1),
            return_attention=True,
        )

    scores = _node_scores(data.num_nodes, attentions)
    mol = Chem.MolFromSmiles(data.smiles)
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
    highlight_atoms = [i for i, score in enumerate(scores) if score > 0]
    highlight_colors = {i: (1.0, 1.0 - 0.8 * score, 1.0 - 0.8 * score) for i, score in enumerate(scores) if score > 0}
    highlight_radii = {i: 0.25 + 0.45 * score for i, score in enumerate(scores) if score > 0}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(drawer.GetDrawingText())
    return scores, output_path
