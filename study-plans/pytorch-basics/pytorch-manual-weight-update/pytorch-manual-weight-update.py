import torch
import torch.nn as nn

def manual_train_step(model, X, y, criterion, lr):
    """
    Returns: loss value as a Python float
    """
    X_arr = torch.tensor(X)
    y_arr = torch.tensor(y)

    y_pred = model(X)

    loss = criterion(y_pred, y_arr)
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param.sub_(lr * param.grad)
            param.grad = None

    return loss.item()
