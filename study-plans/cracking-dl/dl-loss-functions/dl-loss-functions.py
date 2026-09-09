import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    match loss_type:
        case "mse":
            loss = np.mean((y_true - y_pred)**2)
        case "bce":
            y_pred = np.clip(y_pred, 10**-15, 1 - 10**-15)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))
        case "cce":
            maxs = np.max(y_pred, axis=-1, keepdims=True)
            loss = -np.mean(
                y_pred[np.arange(y_pred.shape[0]), y_true] - maxs - np.log(np.sum(np.exp(y_pred - maxs), axis=-1, keepdims=True)))
        case "hinge":
            loss = np.mean(np.maximum(0, 1 - y_true * y_pred))

    return round(loss.item(), 4)