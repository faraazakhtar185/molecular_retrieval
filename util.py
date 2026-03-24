import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from tqdm import tqdm

def collate_contrastive(batch):
    """Custom collate function for contrastive learning batches"""
    anchors = [item['anchor'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negative'] for item in batch]
    
    anchor_batch = Batch.from_data_list(anchors)
    positive_batch = Batch.from_data_list(positives)
    negative_batch = Batch.from_data_list(negatives)
    
    return {
        'anchor': anchor_batch,
        'positive': positive_batch,
        'negative': negative_batch
    }

def contrastive_loss(anchor_proj, positive_proj, negative_proj, temperature=0.1):
    """InfoNCE loss for contrastive learning"""
    
    # Compute similarities (cosine similarity scaled by temperature)
    pos_sim = F.cosine_similarity(anchor_proj, positive_proj, dim=1) / temperature
    neg_sim = F.cosine_similarity(anchor_proj, negative_proj, dim=1) / temperature
    
    # InfoNCE loss: maximize probability of positive pair
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    loss = F.cross_entropy(logits, labels)
    
    return loss

def train_contrastive_model(model, dataloader, optimizer, device, num_epochs=10):
    """Train the contrastive learning model"""
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move to device
            anchor_batch = batch['anchor'].to(device)
            positive_batch = batch['positive'].to(device)
            negative_batch = batch['negative'].to(device)
            
            # Forward pass
            anchor_emb, anchor_proj = model(anchor_batch)
            positive_emb, positive_proj = model(positive_batch)
            negative_emb, negative_proj = model(negative_batch)
            
            # Compute loss
            loss = contrastive_loss(anchor_proj, positive_proj, negative_proj)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')