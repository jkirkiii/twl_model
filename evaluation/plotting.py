from datetime import datetime
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

    # Save predictions with spatial information
    pred_path = save_best_predictions(fold_results, grid_reference, run_id, output_dir)
    log_message(f"\nSaved spatial predictions to: {pred_path}", log_file)

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
    """Create heatmap of RMSE values across the spatial grid, showing only valid cells"""
    # Get valid indices from the first fold result
    valid_indices = fold_results[0]['valid_indices']

    # Initialize arrays for only the valid cells
    n_valid_cells = len(valid_indices)
    squared_errors = np.zeros(n_valid_cells)
    counts = np.zeros(n_valid_cells)

    for result in fold_results:
        val_indices = np.array(result['val_indices'])
        predictions = result['predictions']
        targets = result['targets']

        # Calculate squared errors for each cell
        cell_errors = (predictions - targets) ** 2

        # Accumulate errors for valid cells
        squared_errors += cell_errors.sum(axis=0)
        counts += np.ones(n_valid_cells) * len(val_indices)

    # Calculate RMSE for valid cells
    cell_rmse = np.sqrt(squared_errors / counts)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Only plot valid cells
    scatter = plt.scatter(
        grid_reference.loc[valid_indices, 'FlowElem_xcc'],
        grid_reference.loc[valid_indices, 'FlowElem_ycc'],
        c=cell_rmse,
        cmap='viridis',
        s=2,  # Slightly larger points for better visibility
        alpha=0.8
    )

    plt.colorbar(scatter, label='RMSE (m)')
    plt.title('RMSE Distribution')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')

    # Add statistics annotation
    stats_text = (
        f'Number of cells: {n_valid_cells:,}\n'
        f'RMSE range: {cell_rmse.min():.3f}m - {cell_rmse.max():.3f}m\n'
        f'Mean RMSE: {cell_rmse.mean():.3f}m'
    )
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=8,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

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


def save_best_predictions(fold_results, grid_reference, run_id, output_dir):
    """Save predictions from best model with spatial information."""
    # Find best fold
    best_fold = min(fold_results, key=lambda x: x['metrics']['final_val_loss'])

    # Get predictions and actual values
    predictions = best_fold['predictions']
    targets = best_fold['targets']
    valid_indices = best_fold['valid_indices']

    # Create DataFrame with spatial information for valid cells
    results_df = pd.DataFrame({
        'grid_cell_id': valid_indices,
        'x_coordinate_m': grid_reference.loc[valid_indices, 'FlowElem_xcc'],
        'y_coordinate_m': grid_reference.loc[valid_indices, 'FlowElem_ycc'],
        'predicted_twl_mean_m': predictions.mean(axis=0),
        'predicted_twl_std_m': predictions.std(axis=0),
        'actual_twl_m': targets.mean(axis=0),
        'absolute_error_m': np.abs(predictions.mean(axis=0) - targets.mean(axis=0)),
        'n_samples': predictions.shape[0]
    })

    # Save to CSV in run directory
    output_path = output_dir / f'spatial_predictions_{run_id}.csv'
    results_df.to_csv(output_path, index=False)

    return output_path


def log_message(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')
