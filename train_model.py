import numpy as np
import torch
import sklearn.model_selection
import sklearn.preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from twl_model import TWLModel, WaterLevelDataset, EarlyStopping
from torch import nn, optim
from torch.utils.data import DataLoader


def load_model_inputs(input_file='processed_data/twl_model_inputs.csv'):
    """
    Load and validate model input features from CSV.

    Expected columns include tidal constituents (pM2, pK1, etc.),
    wave characteristics (hs1-3, tp1-3, dir1-3), atmospheric (slp, wdu, wdv),
    and river flows (qsac, qsan).

    Parameters:
        input_file (str): Path to input CSV file

    Returns:
        numpy.ndarray: Array of shape (750, 25) containing model inputs
    """

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} not found")

    expected_columns = [
        'pM2', 'pK1', 'pO1', 'pS2', 'pN2', 'pP1', 'pSA', 'pQ1', 'pK2', 'pSSA',
        'MMSLA',
        'hs1', 'tp1', 'dir1',
        'hs2', 'tp2', 'dir2',
        'hs3', 'tp3', 'dir3',
        'slp', 'wdu', 'wdv',
        'qsac', 'qsan'
    ]

    if len(df) != 750:
        raise ValueError(f"Expected 750 rows, but got {len(df)}")

    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    X = df[expected_columns].values

    if X.shape != (750, 25):
        raise ValueError(f"Expected shape (750, 25), but got {X.shape}")

    return X


def prepare_data(X, y, train_size=0.8, val_size=0.1, batch_size=32):
    """Prepare data for training with proper scaling and splitting"""
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = sklearn.model_selection.train_test_split(
        X, y, train_size=train_size, random_state=42
    )

    val_size_adjusted = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
        X_temp, y_temp, train_size=val_size_adjusted, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create data loaders
    train_dataset = WaterLevelDataset(X_train_scaled, y_train)
    val_dataset = WaterLevelDataset(X_val_scaled, y_val)
    test_dataset = WaterLevelDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler


def train_model(model, train_loader, val_loader, num_epochs=1500,
                learning_rate=0.01, device=None):
    """Train the neural network with early stopping and learning rate scheduling"""
    import time
    from datetime import datetime

    # Set up logging with append mode
    log_file = 'train_log.txt'
    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Training started at {datetime.now()}\n")
        f.write("-" * 50 + "\n")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=200, min_lr=1e-6
    )
    early_stopping = EarlyStopping()

    train_losses = []
    val_losses = []
    epoch_times = []
    training_start_time = time.time()

    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()

        # Calculate average losses and timing
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss)

        if (epoch + 1) % 10 == 0:
            message = (f'Epoch [{epoch + 1}/{num_epochs}], '
                       f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                       f'Time: {epoch_time:.2f}s')
            log_message(message)

        if early_stopping.early_stop:
            message = f"Early stopping triggered at epoch {epoch}"
            log_message(message)
            break

    total_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    mins = int(total_time // 60)
    secs = int(total_time % 60)

    final_stats = (f"\nTraining completed at {datetime.now()}\n"
                   f"Total training time: {mins:02d}:{secs:02d}\n"
                   f"Average epoch time: {avg_epoch_time:.2f} seconds\n"
                   f"Number of epochs completed: {len(epoch_times)}")
    log_message(final_stats)

    return train_losses, val_losses


def evaluate_model(model, test_loader, device=None):
    """Evaluate the model on test data and return performance metrics"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            outputs = model(features)

            total_loss += criterion(outputs, targets).item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    rmse = np.sqrt(avg_loss)

    # Log the test results
    with open('train_log.txt', 'a') as f:
        f.write("\nTest Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write("-" * 20 + "\n")

    return {
        'test_loss': avg_loss,
        'rmse': rmse,
        'predictions': np.array(predictions),
        'actuals': np.array(actuals)
    }


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    input_file = script_dir / 'processed_data/twl_model_inputs.csv'
    twl_file = script_dir / 'processed_data/twls_array.npy'

    try:
        # Load input features
        X = load_model_inputs(input_file)
        with open('train_log.txt', 'a') as f:
            f.write(f"Loaded input data with shape: {X.shape}\n")

        # Load target values (TWL data)
        twl_data = np.load(twl_file)
        twl_data = twl_data.T
        with open('train_log.txt', 'a') as f:
            f.write(f"Loaded TWL data with shape: {twl_data.shape}\n")

        # Prepare data
        train_loader, val_loader, test_loader, scaler = prepare_data(X, twl_data)

        # Initialize model
        model = TWLModel(input_size=25, hidden_size=256, output_size=179269, dropout_rate=0.3)

        # Train model
        train_losses, val_losses = train_model(model, train_loader, val_loader)

        # Evaluate model
        results = evaluate_model(model, test_loader)

    except Exception as e:
        with open('train_log.txt', 'a') as f:
            f.write(f"\nError during model training: {str(e)}\n")
        raise e
