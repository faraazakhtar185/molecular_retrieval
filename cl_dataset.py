import random
from torch.utils.data import Dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from smile_to_graph import SMILESToGraph

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with polymer SMILES"""
    
    def __init__(self, smiles_list, similarity_threshold=0.7):
        self.smiles_list = smiles_list
        self.similarity_threshold = similarity_threshold
        self.graph_converter = SMILESToGraph()
        
        # Pre-compute molecular fingerprints for similarity calculation
        self.fingerprints = []
        self.valid_indices = []

        # Check which molecules are valid  
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                self.fingerprints.append(fp)
                self.valid_indices.append(i)
        
        print(f"Valid molecules: {len(self.valid_indices)} / {len(smiles_list)}")
        
        # Pre-compute positive pairs to speed up training
        self.positive_pairs = self._find_positive_pairs()

    def _get_negative(self, anchor_idx, positive_idx):
        """Find a molecule that's NOT similar to anchor (i.e., not a positive)"""
        anchor_fp = self.fingerprints[anchor_idx]
        
        # Find molecules that are NOT similar (below positive threshold)
        negatives = []
        for i, fp in enumerate(self.fingerprints):
            if i != anchor_idx and i != positive_idx:
                sim = DataStructs.TanimotoSimilarity(anchor_fp, fp)
                if sim < self.similarity_threshold:  # Use same threshold as positives
                    negatives.append(i)
        
        # If we found negatives, use one
        if negatives:
            return random.choice(negatives)
        else:
            # Fallback to random (shouldn't happen often)
            candidates = [i for i in range(len(self.fingerprints)) 
                         if i != anchor_idx and i != positive_idx]
            return random.choice(candidates)

    def _find_positive_pairs(self):
        """Find pairs of similar molecules based on Tanimoto similarity"""
        positive_pairs = []
        
        for i in range(len(self.fingerprints)):
            similarities = []
            for j in range(len(self.fingerprints)):
                if i != j:
                    sim = DataStructs.TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
                    if sim > self.similarity_threshold:
                        similarities.append((j, sim))
            
            if similarities:  # If we found similar molecules
                # Sort by similarity and take the most similar ones
                similarities.sort(key=lambda x: x[1], reverse=True)
                positive_pairs.extend([(i, sim[0]) for sim in similarities[:3]])  # Top 3 most similar
        
        print(f"Found {len(positive_pairs)} positive pairs")
        return positive_pairs

    def __len__(self):
        return len(self.positive_pairs) * 2  # Each positive pair generates one positive + one negative example

    def __getitem__(self, idx):
        """Get a training example (anchor, positive, negative)"""
        pair_idx = idx // 2
        
        if pair_idx >= len(self.positive_pairs):
            pair_idx = random.randint(0, len(self.positive_pairs) - 1)
        
        anchor_idx, positive_idx = self.positive_pairs[pair_idx]
        
        # Get negative 
        negative_idx = self._get_negative(anchor_idx, positive_idx)
        
        # Convert to graphs
        anchor_smiles = self.smiles_list[self.valid_indices[anchor_idx]]
        positive_smiles = self.smiles_list[self.valid_indices[positive_idx]]
        negative_smiles = self.smiles_list[self.valid_indices[negative_idx]]
        
        anchor_graph = self.graph_converter.smiles_to_graph(anchor_smiles)
        positive_graph = self.graph_converter.smiles_to_graph(positive_smiles)
        negative_graph = self.graph_converter.smiles_to_graph(negative_smiles)
        
        return {
            'anchor': anchor_graph,
            'positive': positive_graph,
            'negative': negative_graph
        }