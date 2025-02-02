import numpy as np
import torch


def calculate_rmse(predictions, targets):
    """Calculate root mean squared error"""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_pointwise_rmse(fold_results, n_grid_cells):
    """Calculate RMSE for each grid cell across all predictions"""
    squared_error_sum = np.zeros(n_grid_cells)
    prediction_counts = np.zeros(n_grid_cells)

    for result in fold_results:
        val_indices = np.array(result['val_indices'])
        predictions = result['predictions']
        targets = result['targets']
        squared_errors = (predictions - targets) ** 2

        for i, idx in enumerate(val_indices):
            squared_error_sum += squared_errors[i]
            prediction_counts += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        pointwise_rmse = np.sqrt(squared_error_sum / prediction_counts)

    return np.nan_to_num(pointwise_rmse, nan=0)


def evaluate_fold(model, val_loader, device):
    """Evaluate model performance on a validation fold"""
    model = model.to(device)
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            target = batch['target'].to(device)
            output = model(features)
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    rmse = calculate_rmse(predictions, targets)

    return rmse, predictions, targets
