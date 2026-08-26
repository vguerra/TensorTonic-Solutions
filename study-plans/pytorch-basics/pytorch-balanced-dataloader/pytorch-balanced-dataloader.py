import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns: a DataLoader that oversamples underrepresented classes
    """
    dataset = TensorDataset(features, labels)
    label_counts = torch.bincount(labels)
    class_weights = 1.0 / label_counts
    sample_weights = class_weights[labels]
    print(type(label_counts))
    sampler = WeightedRandomSampler(sample_weights, len(dataset))

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
