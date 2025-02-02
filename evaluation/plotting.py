import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from evaluation.metrics import calculate_pointwise_rmse


def plot_cv_results(fold_results, log_file):
    """Create and save visualization of cross-validation results"""
    # Create unique run identifier with timestamp
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create run-specific directory
    output_dir = Path('cv_results') / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load grid reference
    grid_reference = pd.read_csv('processed_data/grid_reference.csv')

    # Generate all plots
    plot_validation_curves(fold_results, run_id, output_dir)
    plot_rmse_distribution(fold_results, run_id, output_dir)
    plot_rmse_heatmap(fold_results, grid_reference, run_id, output_dir)

    # Save summary statistics
    rmse_values = [result['rmse'] for result in fold_results]
    stats = {
        'mean_rmse': float(np.mean(rmse_values)),
        'std_rmse': float(np.std(rmse_values)),
        'min_rmse': float(np.min(rmse_values)),
        'max_rmse': float(np.max(rmse_values))
    }

    # Save stats to the run directory
    log_message("\nRMSE Statistics for run {}:".format(run_id), log_file)
    for metric, value in stats.items():
        log_message(f"{metric}: {value:.4f}", log_file)

    return run_id


def plot_rmse_heatmap(fold_results, grid_reference, run_id, output_dir):
    """Create heatmap of RMSE values across the spatial grid"""
    n_cells = len(grid_reference)
    squared_errors = np.zeros(n_cells)
    counts = np.zeros(n_cells)

    for result in fold_results:
        val_indices = np.array(result['val_indices'])
        predictions = result['predictions']  # Shape: (n_samples, n_cells)
        targets = result['targets']

        # Calculate squared errors for each cell
        cell_errors = (predictions - targets) ** 2

        # Accumulate errors across folds
        for i, idx in enumerate(val_indices):
            squared_errors += cell_errors[i]
            counts += 1

    # Calculate RMSE where we have predictions
    mask = counts > 0
    cell_rmse = np.zeros(n_cells)
    cell_rmse[mask] = np.sqrt(squared_errors[mask] / counts[mask])

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        grid_reference['FlowElem_xcc'],
        grid_reference['FlowElem_ycc'],
        c=cell_rmse,
        cmap='viridis',
        s=1,
        alpha=0.6
    )

    plt.colorbar(scatter, label='RMSE (m)')
    plt.title('RMSE Distribution Across San Francisco Bay')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')

    plt.text(0.02, 0.98, f'Actual range: {cell_rmse.min():.3f}m - {cell_rmse.max():.3f}m',
             transform=plt.gca().transAxes, fontsize=8)

    plt.savefig(output_dir / f'rmse_heatmap_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_validation_curves(fold_results, run_id, output_dir):
    """Plot validation loss curves for each fold"""
    plt.figure(figsize=(10, 6))

    for fold_idx, result in enumerate(fold_results):
        plt.plot(
            result['val_losses'],
            label=f'Fold {fold_idx + 1}',
            alpha=0.7
        )

    plt.title('Validation Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(output_dir / f'validation_curves_{run_id}.png')
    plt.close()


def plot_rmse_distribution(fold_results, run_id, output_dir):
    """
    Plot distribution of point-wise RMSE values across all grid cells.

    Args:
        fold_results: List of dictionaries containing fold results
        run_id: Unique identifier for this run
        output_dir: Directory to save output plots
    """
    plt.figure(figsize=(12, 6))

    # Calculate point-wise RMSE
    n_grid_cells = fold_results[0]['predictions'].shape[1]  # Number of grid cells
    rmse_values = calculate_pointwise_rmse(fold_results, n_grid_cells)

    # Create histogram
    plt.hist(rmse_values, bins=50, density=True, edgecolor='black')

    # Add statistical annotations
    mean_rmse = np.mean(rmse_values)
    median_rmse = np.median(rmse_values)
    std_rmse = np.std(rmse_values)

    plt.axvline(mean_rmse, color='red', linestyle='dashed',
                label=f'Mean: {mean_rmse:.3f}m')
    plt.axvline(median_rmse, color='green', linestyle='dashed',
                label=f'Median: {median_rmse:.3f}m')

    # Add text box with statistics
    stats_text = (f'Mean: {mean_rmse:.3f}m\n'
                  f'Median: {median_rmse:.3f}m\n'
                  f'Std Dev: {std_rmse:.3f}m\n'
                  f'Min: {np.min(rmse_values):.3f}m\n'
                  f'Max: {np.max(rmse_values):.3f}m')

    plt.text(0.98, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title('Distribution of Point-wise RMSE Across Grid Cells')
    plt.xlabel('RMSE (meters)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig(output_dir / f'pointwise_rmse_distribution_{run_id}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Return statistics for logging
    return {
        'mean_rmse': float(mean_rmse),
        'median_rmse': float(median_rmse),
        'std_rmse': float(std_rmse),
        'min_rmse': float(np.min(rmse_values)),
        'max_rmse': float(np.max(rmse_values))
    }


def log_message(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')
