import pandas as pd
import numpy as np
from scipy.spatial import distance


def find_nearest_grid_cells(observation_points, grid_reference, output_file='observation_points_matched.csv'):
    """
    Find the nearest grid cell for each observation point
    """
    print(f"Finding nearest grid cells for {len(observation_points)} observation points...")

    obs_coords = np.vstack((observation_points['X_m'], observation_points['Y_m'])).T
    grid_coords = np.vstack((grid_reference['FlowElem_xcc'], grid_reference['FlowElem_ycc'])).T

    nearest_indices = []
    distances = []

    for i, obs in enumerate(obs_coords):
        dists = distance.cdist([obs], grid_coords, 'euclidean')[0]
        nearest_idx = np.argmin(dists)

        nearest_indices.append(nearest_idx)
        distances.append(dists[nearest_idx])

    result = observation_points.copy()
    result['nearest_grid_cell_id'] = nearest_indices
    result['distance_to_nearest_m'] = distances

    result['grid_x_m'] = grid_reference.loc[nearest_indices, 'FlowElem_xcc'].values
    result['grid_y_m'] = grid_reference.loc[nearest_indices, 'FlowElem_ycc'].values

    result['x_offset_m'] = result['X_m'] - result['grid_x_m']
    result['y_offset_m'] = result['Y_m'] - result['grid_y_m']

    result.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(f"\nSummary Statistics:")
    print(f"Min distance: {np.min(distances):.2f} m")
    print(f"Max distance: {np.max(distances):.2f} m")
    print(f"Mean distance: {np.mean(distances):.2f} m")
    print(f"Median distance: {np.median(distances):.2f} m")

    return result


if __name__ == "__main__":
    observation_points = pd.read_csv('observation_points.csv')
    grid_reference = pd.read_csv('../processed_data/grid_reference.csv')

    matched_points = find_nearest_grid_cells(
        observation_points,
        grid_reference,
        output_file='../processed_data/observation_points_reference.csv'
    )
