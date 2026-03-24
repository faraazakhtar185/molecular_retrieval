import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_encoder import GraphEncoder

class ContrastiveLearningModel(nn.Module):
    """Complete model for contrastive learning"""
    
    def __init__(self, encoder_params=None):
        super(ContrastiveLearningModel, self).__init__()
        
        if encoder_params is None:
            encoder_params = {}
        
        self.encoder = GraphEncoder(**encoder_params)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_params.get('output_dim', 256), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, data):
        # Get graph embedding
        embedding = self.encoder(data)
        
        # Project to contrastive space
        projection = self.projection_head(embedding)
        
        # L2 normalize for cosine similarity
        projection = F.normalize(projection, dim=1)
        
        return embedding, projection