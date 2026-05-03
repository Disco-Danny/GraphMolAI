from __future__ import annotations

import json
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Data


def compute_atom_importance(model, data: Data, device: torch.device, task_type: str) -> list[dict[str, float | int | str]]:
    """Compute gradient-based atom importance scores for a single molecule."""
    model.eval()

    x = data.x.clone().detach().to(device).requires_grad_(True)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    prediction = model(x, edge_index, edge_attr, batch).view(-1)[0]
    score = torch.sigmoid(prediction) if task_type == "classification" else prediction

    model.zero_grad(set_to_none=True)
    score.backward()

    gradients = x.grad.detach()
    importance = (gradients * x).abs().sum(dim=1)
    if torch.max(importance) > 0:
        importance = importance / torch.max(importance)

    molecule = Chem.MolFromSmiles(data.smiles)
    atom_reports: list[dict[str, float | int | str]] = []
    for atom_index, atom in enumerate(molecule.GetAtoms()):
        atom_reports.append(
            {
                "atom_index": atom_index,
                "atom_symbol": atom.GetSymbol(),
                "importance": float(importance[atom_index].item()),
            }
        )
    return atom_reports


def save_atom_importance(report: list[dict[str, float | int | str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def format_atom_importance(report: list[dict[str, float | int | str]]) -> str:
    lines = ["Atom importance scores:"]
    for item in report:
        lines.append(
            f"  atom {item['atom_index']:>2} ({item['atom_symbol']}): {float(item['importance']):.4f}"
        )
    return "\n".join(lines)
