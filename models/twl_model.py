import torch.nn as nn


class TWLModel(nn.Module):
    """Neural Network for water level prediction with configurable layer sizes"""

    def __init__(self, input_size=25, hidden_sizes=None,
                 output_size=179269, dropout_rate=0.2):
        super(TWLModel, self).__init__()

        # Store initialization parameters
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 256]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # Build layers dynamically
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

    def get_params(self):
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate
        }


class EarlyStopping:
    """
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
