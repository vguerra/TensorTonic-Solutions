import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    """
    Returns: dict with 'train_losses' (list), 'val_losses' (list), 'stopped_epoch' (int, 1-indexed)
    """
    best_val_loss = float('inf')
    no_improv_steps = 0
    avg_train_losses = []
    avg_val_losses = []
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):
        train_losses = []
        val_losses = []

        for inputs, targets in train_loader:
            model.train()
            preds = model(inputs)
            train_loss = criterion(preds, targets)
            train_losses.append(train_loss.item())
            train_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        avg_train_losses.append(sum(train_losses) / len(train_losses))
        
        model.eval()

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_preds = model(val_inputs)
                val_loss = criterion(val_preds, val_targets)
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improv_steps = 0
        else:
            no_improv_steps += 1
    
            if no_improv_steps >= patience:
                stopped_epoch = epoch + 1
                break

    print(avg_train_losses, avg_val_losses, stopped_epoch)
    return {
        'train_losses': avg_train_losses,
        'val_losses': avg_val_losses,
        'stopped_epoch': stopped_epoch
    }

                    
                