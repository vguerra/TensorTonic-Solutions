import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    ac_loss = 0.0
    n_batches = len(dataloader)
    for input, target in dataloader:
        pred = model(input)
        loss = criterion(pred, target)
        ac_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
    return ac_loss / n_batches
        
    
