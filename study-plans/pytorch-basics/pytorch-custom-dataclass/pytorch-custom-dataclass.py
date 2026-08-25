import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    Returns: (features, label) from __getitem__ where features is float32 (D,) and label is float32 (1,)
    """

    def __init__(self, data, label_col):
        self.feature_idxs = [idx for idx in range(len(data[0])) if idx != label_col]
        self.col_idx = label_col
        self.all_data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.all_data.size(0)

    def __getitem__(self, idx):
        return (self.all_data[idx, self.feature_idxs],
                self.all_data[idx, self.col_idx].unsqueeze(0))
