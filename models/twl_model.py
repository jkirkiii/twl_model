import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

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


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)

        # Projection shortcut if dimensions don't match
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = x

        # First transformation
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear1(x)

        # Second transformation
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        # Skip connection
        x += self.shortcut(identity)

        return x


class TWLResNet(nn.Module):
    def __init__(self, input_size=25, hidden_sizes=None, output_size=165590,
                 dropout_rate=0.3, num_res_blocks=5):
        super(TWLResNet, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 512, 768, 512, 256]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_res_blocks = num_res_blocks

        # Input embedding layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_features = hidden_sizes[min(i, len(hidden_sizes) - 1)]
            out_features = hidden_sizes[min(i + 1, len(hidden_sizes) - 1)]
            self.res_blocks.append(ResidualBlock(in_features, out_features, dropout_rate))

        # Final output projection layer
        self.output_layer = nn.Linear(hidden_sizes[min(num_res_blocks, len(hidden_sizes) - 1)], output_size)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.output_layer(x)
        return x

    def get_params(self):
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'num_res_blocks': self.num_res_blocks
        }


class TWLResNetFeatModel(nn.Module):
    def __init__(self, input_size=25, hidden_sizes=None, output_size=165590,
                 dropout_rate=0.3, num_res_blocks=5):
        super(TWLResNetFeatModel, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 512, 384, 256]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_res_blocks = num_res_blocks

        self.tidal_pathway = nn.Sequential(
            nn.Linear(10, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.wave_pathway = nn.Sequential(
            nn.Linear(9, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.atm_pathway = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.river_pathway = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.msl_pathway = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Merge layer to combine all pathways
        merged_size = 128 + 128 + 64 + 32 + 16

        # Main backbone with residual connections
        self.input_layer = nn.Sequential(
            nn.Linear(merged_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_features = hidden_sizes[min(i, len(hidden_sizes) - 1)]
            out_features = hidden_sizes[min(i + 1, len(hidden_sizes) - 1)]
            self.res_blocks.append(ResidualBlock(in_features, out_features, dropout_rate))

        # Output projection layer
        self.output_layer = nn.Linear(hidden_sizes[min(num_res_blocks, len(hidden_sizes) - 1)], output_size)

    def forward(self, x):
        # Split input features into different physical processes
        tidal_features = x[:, :10]
        wave_features = x[:, 10:19]
        atm_features = x[:, 19:22]
        river_features = x[:, 22:24]
        msl_features = x[:, 24:25]

        tidal_out = self.tidal_pathway(tidal_features)
        wave_out = self.wave_pathway(wave_features)
        atm_out = self.atm_pathway(atm_features)
        river_out = self.river_pathway(river_features)
        msl_out = self.msl_pathway(msl_features)

        # Merge pathways
        merged = torch.cat([tidal_out, wave_out, atm_out, river_out, msl_out], dim=1)

        # Input embedding
        x = self.input_layer(merged)

        # Process through residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output projection
        x = self.output_layer(x)
        return x

    def get_params(self):
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'num_res_blocks': self.num_res_blocks
        }


class SpatiallyAwareTWLModel(nn.Module):
    """
    <<WIP>>
    Spatially-aware model for predicting water levels across a grid
    Explicitly models the interaction between environmental conditions and spatial location
    """

    def __init__(self, input_size=25, hidden_sizes=None, output_size=165590,
                 spatial_embedding_dim=16, dropout_rate=0.3):
        super(SpatiallyAwareTWLModel, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.spatial_embedding_dim = spatial_embedding_dim

        # Environmental feature encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU()
        )

        # Register buffer for spatial coordinates (will be filled later)
        self.register_buffer('spatial_coords', torch.zeros(output_size, 2))
        self.register_buffer('bed_levels', torch.zeros(output_size, 1))

        # Spatial feature encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, spatial_embedding_dim),  # x, y, bed_level
            nn.ReLU(),
            nn.Linear(spatial_embedding_dim, spatial_embedding_dim),
            nn.ReLU()
        )

        # Decoder for both environmental and spatial features
        combined_dim = hidden_sizes[1] + spatial_embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[2], 1)
        )

    # def forward(self, x):
    #     batch_size = x.shape[0]
    #
    #     env_features = self.env_encoder(x)
    #
    #     spatial_input = torch.cat([self.spatial_coords, self.bed_levels], dim=1)
    #
    #     spatial_features = self.spatial_encoder(spatial_input)
    #
    #     env_features = env_features.unsqueeze(1).expand(-1, self.output_size, -1)
    #     spatial_features = spatial_features.unsqueeze(0).expand(batch_size, -1, -1)
    #
    #     combined = torch.cat([env_features, spatial_features], dim=2)
    #     output = self.decoder(combined).squeeze(-1)
    #
    #     return output

    def forward(self, x):
        batch_size = x.shape[0]
        env_features = self.env_encoder(x)

        chunk_size = 5000
        outputs = []

        for i in range(0, self.output_size, chunk_size):
            end_idx = min(i + chunk_size, self.output_size)
            chunk_coords = self.spatial_coords[i:end_idx]
            chunk_bed = self.bed_levels[i:end_idx]

            spatial_input = torch.cat([chunk_coords, chunk_bed], dim=1)
            spatial_features = self.spatial_encoder(spatial_input)

            env_expanded = env_features.unsqueeze(1).expand(-1, end_idx - i, -1)
            spatial_expanded = spatial_features.unsqueeze(0).expand(batch_size, -1, -1)

            combined = torch.cat([env_expanded, spatial_expanded], dim=2)
            chunk_output = self.decoder(combined).squeeze(-1)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=1)

    def load_spatial_data(self, grid_reference, valid_indices):
        # Extract coordinates
        x_coords = grid_reference.loc[valid_indices, 'x_km'].values
        y_coords = grid_reference.loc[valid_indices, 'y_km'].values
        bed_levels = grid_reference.loc[valid_indices, 'bed_level'].values

        # Normalize
        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())

        # Normalize
        bed_min, bed_max = bed_levels.min(), bed_levels.max()
        bed_norm = (bed_levels - bed_min) / (bed_max - bed_min)

        self.x_range = (x_coords.min(), x_coords.max())
        self.y_range = (y_coords.min(), y_coords.max())
        self.bed_range = (bed_min, bed_max)

        # Create coordinate tensor
        coords = torch.tensor(np.stack([x_norm, y_norm], axis=1), dtype=torch.float32)
        self.spatial_coords.copy_(coords)

        # Create bed level tensor
        bed_tensor = torch.tensor(bed_norm.reshape(-1, 1), dtype=torch.float32)
        self.bed_levels.copy_(bed_tensor)

        self.output_size = len(valid_indices)

        print(f"Loaded spatial data for {self.output_size} valid grid cells")

    def get_params(self):
        """Return model parameters for saving/loading"""
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'spatial_embedding_dim': self.spatial_embedding_dim,
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
