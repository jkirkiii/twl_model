from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from evaluation.metrics import calculate_pointwise_rmse


def plot_cv_results(fold_results, log_file, observation_matches_file=None):
    """Create and save visualization of cross-validation results"""
    # Create unique run identifier with timestamp
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create run-specific directory
    output_dir = Path('cv_results') / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load grid reference
    grid_reference = pd.read_csv('processed_data/grid_reference.csv')

    # Generate standard plots
    plot_validation_curves(fold_results, run_id, output_dir)
    plot_rmse_distribution(fold_results, run_id, output_dir)
    plot_rmse_heatmap(fold_results, grid_reference, run_id, output_dir)

    # If observation matches file provided, generate observation-specific visualizations
    if observation_matches_file and Path(observation_matches_file).exists():
        try:
            observation_matches = pd.read_csv(observation_matches_file)
            log_message(f"Generating observation point visualizations from {observation_matches_file}", log_file)
            plot_rmse_heatmap_with_observations(fold_results, grid_reference, observation_matches, run_id, output_dir)
        except Exception as e:
            log_message(f"Error generating observation visualizations: {str(e)}", log_file)

    # Save predictions with spatial information
    pred_path = save_best_predictions(fold_results, grid_reference, run_id, output_dir)

    # Add observation group information if available
    if observation_matches_file and Path(observation_matches_file).exists():
        add_observation_groups(pred_path, observation_matches_file)

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
    cell_rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))

    # Create DataFrame with spatial information for valid cells
    results_df = pd.DataFrame({
        'grid_cell_id': valid_indices,
        'x_coordinate_m': grid_reference.loc[valid_indices, 'FlowElem_xcc'],
        'y_coordinate_m': grid_reference.loc[valid_indices, 'FlowElem_ycc'],
        'predicted_twl_mean_m': predictions.mean(axis=0),
        'predicted_twl_std_m': predictions.std(axis=0),
        'actual_twl_m': targets.mean(axis=0),
        'absolute_error_m': np.abs(predictions.mean(axis=0) - targets.mean(axis=0)),
        'rmse_m': cell_rmse,
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


def add_observation_groups(predictions_file, observation_matches_file):
    """
    Add observation group information to an existing predictions CSV file.

    Args:
        predictions_file: Path to the predictions CSV file
        observation_matches_file: Path to the observation matches CSV file
    """
    # Load the files
    predictions = pd.read_csv(predictions_file)
    observations = pd.read_csv(observation_matches_file)

    # Create a simpler dataframe from observations with just the needed columns
    obs_map = observations[['nearest_grid_cell_id', 'group_name', 'name']]
    obs_map['observation_group'] = obs_map['group_name'] + ' - ' + obs_map['name']
    obs_map = obs_map[['nearest_grid_cell_id', 'observation_group']]

    # Merge with predictions on grid_cell_id
    result = pd.merge(
        predictions,
        obs_map.rename(columns={'nearest_grid_cell_id': 'grid_cell_id'}),
        on='grid_cell_id',
        how='left'  # Keep all predictions, add observation info where available
    )

    # Fill NaN values for cells without observation data
    result['observation_group'] = result['observation_group'].fillna('none')

    # Save the updated predictions
    result.to_csv(predictions_file, index=False)
    print(f"Added observation group information to {predictions_file}")

    return predictions_file


def plot_rmse_heatmap_with_observations(fold_results, grid_reference, observation_matches, run_id, output_dir):
    """
    Create heatmap of RMSE values across the spatial grid, showing observation points by group

    Args:
        fold_results: List of fold result dictionaries
        grid_reference: DataFrame with grid cell reference information
        observation_matches: DataFrame with matched observation points and grid cells
        run_id: Unique identifier for the current run
        output_dir: Directory to save output files
    """
    # Get valid indices from the first fold result
    valid_indices = fold_results[0]['valid_indices']

    # Calculate RMSE for each grid cell across all validation sets
    n_valid_cells = len(valid_indices)
    cell_rmse = calculate_pointwise_rmse(fold_results, n_valid_cells)

    # Determine global extents from all valid cells for consistency across plots
    x_coords = grid_reference.loc[valid_indices, 'FlowElem_xcc']
    y_coords = grid_reference.loc[valid_indices, 'FlowElem_ycc']

    global_extents = {
        'x_min': x_coords.min(),
        'x_max': x_coords.max(),
        'y_min': y_coords.min(),
        'y_max': y_coords.max()
    }

    # Add a small padding (2% of range)
    x_pad = (global_extents['x_max'] - global_extents['x_min']) * 0.02
    y_pad = (global_extents['y_max'] - global_extents['y_min']) * 0.02

    global_extents['x_min'] -= x_pad
    global_extents['x_max'] += x_pad
    global_extents['y_min'] -= y_pad
    global_extents['y_max'] += y_pad

    # Create the main heatmap
    plt.figure(figsize=(15, 10))

    # Plot background for all valid cells
    scatter = plt.scatter(
        x_coords,
        y_coords,
        c=cell_rmse,
        cmap='viridis',
        s=2,
        alpha=0.8
    )

    plt.colorbar(scatter, label='RMSE (m)')
    plt.title('RMSE Distribution with Observation Points')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')

    # Set the global extents for the main heatmap
    plt.xlim(global_extents['x_min'], global_extents['x_max'])
    plt.ylim(global_extents['y_min'], global_extents['y_max'])

    # Plot observation points by group with distinctive markers
    groups = observation_matches['group_name'].unique()
    markers = ['o', 's', '^', 'D', 'p', '*', 'X', 'P']  # Different marker styles

    for i, group in enumerate(groups):
        group_points = observation_matches[observation_matches['group_name'] == group]
        marker = markers[i % len(markers)]  # Cycle through markers if more groups than markers

        plt.scatter(
            group_points['grid_x_m'],
            group_points['grid_y_m'],
            marker=marker,
            s=80,
            facecolors='none',
            edgecolors='white',
            alpha=0.5,
            linewidth=2,
            label=f'{group} ({len(group_points)})'
        )

        # Add a second layer with same marker but different color to make them stand out
        plt.scatter(
            group_points['grid_x_m'],
            group_points['grid_y_m'],
            marker=marker,
            s=50,
            facecolors='none',
            edgecolors='black',
            linewidth=1,
            alpha=0.5
        )

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

    plt.legend(loc='lower right', fontsize=8)
    plt.savefig(output_dir / f'rmse_heatmap_with_observations_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate heatmaps for each group
    for i, group in enumerate(groups):
        group_points = observation_matches[observation_matches['group_name'] == group]
        group_indices = group_points['nearest_grid_cell_id'].values

        # Map grid cell IDs to indices in the valid_indices array
        valid_indices_dict = {cell_id: idx for idx, cell_id in enumerate(valid_indices)}
        group_cell_rmse_indices = [valid_indices_dict.get(cell_id) for cell_id in group_indices
                                   if cell_id in valid_indices_dict]

        # 1. First create heatmap with all cells but focusing on this group's area
        plt.figure(figsize=(15, 10))

        # Plot background for context (all valid cells)
        plt.scatter(
            grid_reference.loc[valid_indices, 'FlowElem_xcc'],
            grid_reference.loc[valid_indices, 'FlowElem_ycc'],
            c=cell_rmse,
            cmap='viridis',
            s=2,
            alpha=0.5
        )

        # Highlight the specific group points
        plt.scatter(
            group_points['grid_x_m'],
            group_points['grid_y_m'],
            marker=markers[i % len(markers)],
            s=50,
            facecolors='none',
            edgecolors='red',
            linewidth=2,
            alpha=0.5,
            label=f'{group} ({len(group_points)})'
        )

        # Calculate statistics for this group's points
        if group_cell_rmse_indices:
            group_rmse_values = cell_rmse[group_cell_rmse_indices]
            group_stats = (
                f'Group: {group}\n'
                f'Points: {len(group_points)}\n'
                f'Valid points: {len(group_cell_rmse_indices)}\n'
                f'RMSE range: {group_rmse_values.min():.3f}m - {group_rmse_values.max():.3f}m\n'
                f'Mean RMSE: {group_rmse_values.mean():.3f}m'
            )
        else:
            group_stats = f'Group: {group}\nNo valid points found in model domain'

        plt.title(f'RMSE Distribution for {group} Observation Points')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.colorbar(label='RMSE (m)')

        plt.text(0.02, 0.98, group_stats,
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.legend(loc='lower right')

        # Then in the group-specific plots, use the global extents:
        plt.xlim(global_extents['x_min'], global_extents['x_max'])
        plt.ylim(global_extents['y_min'], global_extents['y_max'])

        plt.savefig(output_dir / f'rmse_heatmap_{group.replace(" ", "_")}_{run_id}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Now create a new heatmap that only shows the cells in this group
        if group_cell_rmse_indices:
            plt.figure(figsize=(15, 10))

            # Get the grid cells and RMSE values just for this group
            group_cell_ids = [valid_indices[idx] for idx in group_cell_rmse_indices]
            group_x = grid_reference.loc[group_cell_ids, 'FlowElem_xcc']
            group_y = grid_reference.loc[group_cell_ids, 'FlowElem_ycc']
            group_rmse = cell_rmse[group_cell_rmse_indices]

            # Plot only the cells in this group
            scatter = plt.scatter(
                group_x,
                group_y,
                c=group_rmse,
                cmap='viridis',
                s=10,  # Larger points since we have fewer
                alpha=1.0
            )

            # # Highlight the specific observation points
            # plt.scatter(
            #     group_points['grid_x_m'],
            #     group_points['grid_y_m'],
            #     marker=markers[i % len(markers)],
            #     s=100,
            #     facecolors='none',
            #     edgecolors='red',
            #     linewidth=2,
            #     label=f'{group} ({len(group_points)})'
            # )

            plt.colorbar(scatter, label='RMSE (m)')
            plt.title(f'RMSE Distribution for {group} Cells Only')
            plt.xlabel('Easting (m)')
            plt.ylabel('Northing (m)')

            # Use the same statistics as before
            plt.text(0.02, 0.98, group_stats,
                     transform=plt.gca().transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.8))

            # plt.legend(loc='lower right')

            plt.xlim(global_extents['x_min'], global_extents['x_max'])
            plt.ylim(global_extents['y_min'], global_extents['y_max'])

            plt.savefig(output_dir / f'rmse_heatmap_{group.replace(" ", "_")}_only_{run_id}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    # Create RMSE histograms by group
    plot_rmse_histograms_by_group(cell_rmse, valid_indices, observation_matches, run_id, output_dir)

    return


def plot_rmse_histograms_by_group(cell_rmse, valid_indices, observation_matches, run_id, output_dir):
    """
    Create histograms of RMSE values for each observation group

    Args:
        cell_rmse: Array of RMSE values for each valid grid cell
        valid_indices: Array of valid grid cell indices
        observation_matches: DataFrame with matched observation points
        run_id: Unique identifier for this run
        output_dir: Directory to save output files
    """
    # Create mapping from grid_cell_id to position in cell_rmse array
    valid_indices_dict = {cell_id: idx for idx, cell_id in enumerate(valid_indices)}

    # Get all groups and prepare figure
    groups = observation_matches['group_name'].unique()

    # Determine a common x-axis range for all histograms
    # Start with the overall range from all cells
    x_min = np.min(cell_rmse)
    x_max = np.max(cell_rmse)

    # Calculate bin edges for consistent binning across all histograms
    num_bins = 50
    bin_edges = np.linspace(x_min, x_max, num_bins + 1)

    # Overall histogram
    plt.figure(figsize=(12, 6))
    plt.hist(cell_rmse, bins=bin_edges, alpha=0.7, edgecolor='black', label='All grid cells')

    plt.title('RMSE Distribution - All Grid Cells')
    plt.xlabel('RMSE (meters)')
    plt.ylabel('Count')

    # Add statistics annotation
    stats_text = (
        f'Cells: {len(cell_rmse)}\n'
        f'Mean: {np.mean(cell_rmse):.3f}m\n'
        f'Median: {np.median(cell_rmse):.3f}m\n'
        f'Min: {np.min(cell_rmse):.3f}m\n'
        f'Max: {np.max(cell_rmse):.3f}m'
    )

    plt.text(0.98, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'rmse_histogram_all_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate histograms for each group
    for group in groups:
        group_points = observation_matches[observation_matches['group_name'] == group]
        group_indices = group_points['nearest_grid_cell_id'].values

        # Map grid cell IDs to indices in the valid_indices array
        group_cell_rmse_indices = [valid_indices_dict.get(cell_id) for cell_id in group_indices
                                   if cell_id in valid_indices_dict]

        if not group_cell_rmse_indices:
            continue  # Skip groups with no valid points

        group_rmse = cell_rmse[group_cell_rmse_indices]

        plt.figure(figsize=(12, 6))

        # Plot group distribution with the same bin edges as the overall histogram
        plt.hist(group_rmse, bins=bin_edges, alpha=0.7, color='blue', edgecolor='black',
                 label=f'{group} ({len(group_rmse)} points)')

        plt.title(f'RMSE Distribution - {group} Observation Points')
        plt.xlabel('RMSE (meters)')
        plt.ylabel('Count')

        # Use the same x-axis limits for consistent comparison
        plt.xlim(x_min, x_max)

        # Add statistics annotation
        group_stats = (
            f'Group: {group}\n'
            f'Points: {len(group_rmse)}\n'
            f'Mean: {np.mean(group_rmse):.3f}m\n'
            f'Median: {np.median(group_rmse):.3f}m\n'
            f'Min: {np.min(group_rmse):.3f}m\n'
            f'Max: {np.max(group_rmse):.3f}m'
        )

        plt.text(0.98, 0.95, group_stats,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'rmse_histogram_{group.replace(" ", "_")}_{run_id}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
