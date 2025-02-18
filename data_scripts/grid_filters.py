import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_cell_connectivity(file_path, easting_threshold=610000):
    ds = xr.open_dataset(file_path)

    x_coords = ds['FlowElem_xcc'].values
    y_coords = ds['FlowElem_ycc'].values

    flow_links = ds['FlowLink'].values
    flow_link_types = ds['FlowLinkType'].values
    bed_levels = ds['FlowElem_bl'].values

    print(f"Flow links range: {flow_links.min()} to {flow_links.max()} (before adjustment)")

    flow_links = flow_links - 1
    n_elements = len(x_coords)

    adjacency = [{
        'neighbors': set(),
        'link_types': set(),
        'is_1d': False,
        'depth': bed_levels[i]
    } for i in range(n_elements)]

    # Process flow links and their types
    for i, (link, link_type) in enumerate(zip(flow_links, flow_link_types)):
        elem1, elem2 = link
        if elem1 >= 0 and elem2 >= 0 and elem1 < n_elements and elem2 < n_elements:
            adjacency[elem1]['neighbors'].add(elem2)
            adjacency[elem2]['neighbors'].add(elem1)
            adjacency[elem1]['link_types'].add(link_type)
            adjacency[elem2]['link_types'].add(link_type)

            # Mark cells connected by 1D links
            if link_type in [1, 3, 4]:
                adjacency[elem1]['is_1d'] = True
                adjacency[elem2]['is_1d'] = True

    elements_to_exclude = {}
    cell_characteristics = {}

    for i in range(n_elements):
        cell_info = adjacency[i]
        n_neighbors = len(cell_info['neighbors'])
        depth = cell_info['depth']

        cell_characteristics[i] = {
            'x': x_coords[i],
            'y': y_coords[i],
            'n_neighbors': n_neighbors,
            'link_types': cell_info['link_types'],
            'is_1d': cell_info['is_1d'],
            'depth': depth
        }

        # Geographic threshold check
        if x_coords[i] > easting_threshold:
            elements_to_exclude[i] = "East of geographic threshold"
            continue

        if cell_info['is_1d']:
            if n_neighbors <= 2:
                elements_to_exclude[i] = "Tributary (1D channel)"

    print(f"\nProcessing complete:")
    print(f"Total elements: {n_elements}")
    print(f"Elements to exclude: {len(elements_to_exclude)}")

    ds.close()
    return elements_to_exclude, cell_characteristics


def visualize_excluded_cells(cell_characteristics, excluded_cells, output_file):
    plt.figure(figsize=(15, 10))

    categories = {
        "East of geographic threshold": 'blue',
        "Tributary (1D channel)": 'red',
        "Isolated/terminal element": 'purple'
    }

    # Plot background for context
    x_coords = [char['x'] for char in cell_characteristics.values()]
    y_coords = [char['y'] for char in cell_characteristics.values()]

    plt.scatter(x_coords, y_coords, c='lightgray', s=1, alpha=0.3, label='Included cells')

    # Plot excluded cells by category
    for category, color in categories.items():
        indices = [i for i, r in excluded_cells.items() if r == category]
        if indices:
            plt.scatter(
                [cell_characteristics[i]['x'] for i in indices],
                [cell_characteristics[i]['y'] for i in indices],
                c=color, s=1, alpha=0.7, label=category
            )

    plt.title('Grid Cell Analysis Results')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.legend()
    plt.axvline(x=610000, color='black', linestyle='--')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    input_file = Path(r"Z:\School\Capstone\Data\Result_map\sim_1\SFBD_map.nc")
    output_dir = Path(r"Z:\School\Capstone\twl_model\processed_data")
    output_file = output_dir / "cells_to_exclude4.csv"
    plot_file = output_dir / "excluded_cells_visualization4.png"

    try:
        print("Analyzing mesh structure...")
        excluded_cells, cell_characteristics = analyze_cell_connectivity(input_file)

        # Save results
        df = pd.DataFrame({
            'grid_cell_id': list(excluded_cells.keys()),
            'reason': list(excluded_cells.values())
        })
        df.to_csv(output_file, index=False)

        # Create visualization
        visualize_excluded_cells(cell_characteristics, excluded_cells, plot_file)

        # Print summary statistics
        print(f"\nFound {len(excluded_cells)} elements to exclude:")
        for reason in set(excluded_cells.values()):
            count = sum(1 for r in excluded_cells.values() if r == reason)
            print(f"- {reason}: {count} cells")

        # Print connectivity statistics
        n_neighbors = [char['n_neighbors'] for char in cell_characteristics.values()]
        if n_neighbors:
            print(f"\nConnectivity statistics:")
            print(f"Average neighbors: {np.mean(n_neighbors):.1f}")
            print(f"Min neighbors: {min(n_neighbors)}")
            print(f"Max neighbors: {max(n_neighbors)}")

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise


if __name__ == "__main__":
    main()
