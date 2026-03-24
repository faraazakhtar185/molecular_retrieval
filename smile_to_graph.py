import torch
from rdkit import Chem
from torch_geometric.data import Data

class SMILESToGraph:
    """Convert SMILES strings to PyTorch Geometric graph objects"""

    def __init__(self):
        # Atomic features we'll use
        self.atom_features = [
            'atomic_num', 'degree', 'formal_charge', 'hybridization',
            'is_aromatic', 'is_ring'
        ]

    def get_atom_features(self, atom):
        """Extract features for a single atom"""
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing())
        ]
        return features

    def smiles_to_graph(self, smiles):
        """Convert SMILES to PyTorch Geometric Data object"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))

        # Get bond information (edge indices and features)
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Add both directions for undirected graph
            edge_indices.extend([[i, j], [j, i]])

            bond_type = bond.GetBondType()
            bond_features = [
                int(bond_type == Chem.rdchem.BondType.SINGLE),
                int(bond_type == Chem.rdchem.BondType.DOUBLE),
                int(bond_type == Chem.rdchem.BondType.TRIPLE),
                int(bond_type == Chem.rdchem.BondType.AROMATIC),
                int(bond.IsInRing())
            ]
            edge_features.extend([bond_features, bond_features])

        # Convert to tensors
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)