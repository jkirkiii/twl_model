import torch
from torch.utils.data import Dataset


class WaterLevelDataset(Dataset):
    """Custom Dataset for loading water level prediction data"""

    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'features': self.X[idx],
            'target': self.y[idx]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
