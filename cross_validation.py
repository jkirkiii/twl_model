import time
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from pathlib import Path
from models.twl_model import EarlyStopping
from torch import nn, optim
from datetime import datetime
from models.dataset import WaterLevelDataset
from evaluation.metrics import evaluate_fold
from evaluation.plotting import log_message


def train_model(model, train_loader, val_loader, num_epochs=1500,
                learning_rate=0.001, device=None):
    """Train the neural network with early stopping and learning rate scheduling"""

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
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )

    early_stopping = EarlyStopping(patience=150)

    train_losses = []
    val_losses = []
    epoch_times = []
    train_rmse = []
    val_rmse = []
    training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []

        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate average losses and timing
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        current_train_rmse = np.sqrt(np.mean((np.array(train_predictions) - np.array(train_targets)) ** 2))
        current_val_rmse = np.sqrt(np.mean((np.array(val_predictions) - np.array(val_targets)) ** 2))

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmse.append(current_train_rmse)
        val_rmse.append(current_val_rmse)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss)

        if (epoch + 1) % 10 == 0:
            message = (f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Train RMSE: {current_train_rmse:.4f}, '
                      f'Val RMSE: {current_val_rmse:.4f}')
            log_message(message, log_file)

        if early_stopping.early_stop:
            message = f"Early stopping triggered at epoch {epoch}"
            log_message(message, log_file)
            break

    total_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    mins = int(total_time // 60)
    secs = int(total_time % 60)

    final_stats = (f"\nTraining completed at {datetime.now()}\n"
                   f"Total training time: {mins:02d}:{secs:02d}\n"
                   f"Average epoch time: {avg_epoch_time:.2f} seconds\n"
                   f"Number of epochs completed: {len(epoch_times)}")
    log_message(final_stats, log_file)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse
    }


def cross_validate_model(X, y, model_class, grid_reference=None, valid_indices=None, n_splits=5, **model_params):
    """
    Perform k-fold cross validation of the TWL model with detailed logging.
    """

    # Initialize logging
    log_file = 'train_log.txt'
    cv_results_dir = Path('cv_results')
    cv_results_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Starting {n_splits}-fold cross validation at {datetime.now()}\n")
        f.write(f"Data shapes: X={X.shape}, y={y.shape}\n")
        f.write(f"Model parameters:\n")
        f.write(f"  Input size: {model_params['input_size']}\n")
        f.write(f"  Hidden layer sizes: {model_params['hidden_sizes']}\n")
        f.write(f"  Output size: {model_params['output_size']}\n")
        f.write(f"  Dropout rate: {model_params['dropout_rate']}\n")
        f.write("-" * 80 + "\n")

    # Initialize K-Fold cross validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=104)
    fold_results = []
    rmse_scores = []

    # Track metrics across folds
    all_val_losses = []
    all_train_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_start_time = datetime.now()

        with open(log_file, 'a') as f:
            f.write(f"\nStarting Fold {fold + 1}/{n_splits}\n")
            f.write(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}\n")

        # Initialize model and scaler for this fold
        model = model_class(**model_params)

        # Load spatial data if using a spatial model
        if hasattr(model, 'load_spatial_data') and grid_reference is not None and valid_indices is not None:
            model.load_spatial_data(grid_reference, valid_indices)

        scaler = StandardScaler()

        # Scale the features
        X_train_fold = scaler.fit_transform(X[train_idx])
        X_val_fold = scaler.transform(X[val_idx])

        train_loader = DataLoader(
            WaterLevelDataset(X_train_fold, y[train_idx]),
            batch_size=8,
            shuffle=True
        )

        val_loader = DataLoader(
            WaterLevelDataset(X_val_fold, y[val_idx]),
            batch_size=8,
            shuffle=False
        )

        # Train the model for this fold
        train_results = train_model(model, train_loader, val_loader, num_epochs=1500)
        train_losses = train_results['train_losses']
        val_losses = train_results['val_losses']

        fold_rmse, fold_predictions, fold_targets = evaluate_fold(model, val_loader, device)

        rmse_scores.append(fold_rmse)

        log_message(f"\nFold {fold + 1} RMSE: {fold_rmse:.4f} m", log_file)

        # Log fold completion and results
        fold_end_time = datetime.now()
        fold_duration = (fold_end_time - fold_start_time).total_seconds() / 60

        fold_metrics = {
            'fold': fold + 1,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'duration_minutes': fold_duration,
            'rmse': fold_rmse
        }

        with open(log_file, 'a') as f:
            f.write(f"\nFold {fold + 1} completed in {fold_duration:.2f} minutes\n")
            f.write(f"Final training loss: {float(fold_metrics['final_train_loss']):.6f}\n")
            f.write(f"Final validation loss: {float(fold_metrics['final_val_loss']):.6f}\n")
            f.write(f"RMSE: {fold_metrics['rmse']:.6f}\n")
            f.write("-" * 40 + "\n")

        # Save detailed fold results
        fold_results.append({
            'metrics': fold_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model': model,
            'scaler': scaler,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(),
            'rmse': fold_rmse,
            'predictions': fold_predictions,
            'targets': fold_targets
        })

        # Save fold-specific results
        np.save(
            cv_results_dir / f'fold_{fold + 1}_losses.npy',
            {'train': train_losses, 'val': val_losses}
        )

        all_val_losses.append(fold_metrics['final_val_loss'])
        all_train_losses.append(fold_metrics['final_train_loss'])

    # Compute and log final cross-validation statistics
    cv_stats = {
        'mean_val_loss': np.mean(all_val_losses),
        'std_val_loss': np.std(all_val_losses),
        'mean_train_loss': np.mean(all_train_losses),
        'std_train_loss': np.std(all_train_losses),
        'mean_rmse': np.mean(rmse_scores),
        'std_rmse': np.std(rmse_scores)
    }

    with open(log_file, 'a') as f:
        f.write("\nCross-validation completed at {datetime.now()}\n")
        f.write("Final Statistics:\n")
        f.write(f"Mean validation loss: {cv_stats['mean_val_loss']:.6f} ± {cv_stats['std_val_loss']:.6f}\n")
        f.write(f"Mean training loss: {cv_stats['mean_train_loss']:.6f} ± {cv_stats['std_train_loss']:.6f}\n")
        f.write(f"Mean RMSE: {cv_stats['mean_rmse']:.6f} ± {cv_stats['std_rmse']:.6f}\n")
        f.write("=" * 80 + "\n")

    return fold_results, cv_stats
