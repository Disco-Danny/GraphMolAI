from __future__ import annotations

from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert a SMILES string to an ECFP bit vector."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string encountered: {smiles}")

    bit_vector = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def batch_smiles_to_ecfp(
    smiles_list: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Convert a sequence of SMILES strings to a 2D fingerprint matrix."""
    fingerprints = [smiles_to_ecfp(smiles, radius=radius, n_bits=n_bits) for smiles in smiles_list]
    if not fingerprints:
        return np.empty((0, n_bits), dtype=np.float32)
    return np.vstack(fingerprints)
