import torch
import torch.nn as nn
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


class TWLModel(nn.Module):
    """Neural Network for water level prediction

    Basic architecture with three dense layers, batch normalization,
    and dropout for regularization.
    """

    def __init__(self, input_size=25, hidden_size=64, output_size=1000, dropout_rate=0.2):
        super(TWLModel, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size,
                          hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for i in range(3)
        ])

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class EarlyStopping:
    """Early stopping to prevent overfitting

    Tracks validation loss and stops training if no improvement
    is seen after a specified number of epochs.
    """

    def __init__(self, patience=200, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
