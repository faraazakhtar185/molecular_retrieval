
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GraphEncoder(nn.Module):
    """Graph Neural Network encoder for molecular graphs"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=256, num_layers=3):
        super(GraphEncoder, self).__init__()
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
        
        # Last layer (no concatenation)
        self.gat_layers.append(GATConv(hidden_dim * 4, output_dim, heads=1, concat=False))
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:  # No activation on last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        graph_embedding = global_mean_pool(x, batch)
        
        return graph_embedding