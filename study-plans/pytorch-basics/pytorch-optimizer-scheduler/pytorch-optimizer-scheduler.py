import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """

    avg_losses = []
    lrs = []
    for epoch in range(num_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        losses = []
        for inputs, targets in dataloader:
            preds = model(inputs)
            loss = criterion(preds, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_losses.append(sum(losses)/len(dataloader))

    return {
        'losses': avg_losses,
        'lrs': lrs
    }
