import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


def extract_twl_data(file_path, simulation_id):
    """
    Extract Total Water Levels (TWL) from a single simulation file,
    using native Delft3D FM element indexing.

    Parameters:
        file_path (str): Path to the SFBD_map.nc file
        simulation_id (int): Identifier for this simulation run

    Returns:
        pandas.DataFrame: DataFrame containing grid coordinates and TWL values
    """
    ds = xr.open_dataset(file_path)

    # Use native element indices
    native_indices = np.arange(ds.dims['nFlowElem'])

    # Extract coordinates and TWL values
    flow_elem_xcc = ds['FlowElem_xcc'].values
    flow_elem_ycc = ds['FlowElem_ycc'].values
    twl_values = ds['s1'].isel(time=1).values

    # Set any negative or very small values to 0
    twl_values = np.where(twl_values <= 1e-5, 0, twl_values)

    # Create DataFrame with native element indexing
    results_df = pd.DataFrame({
        'grid_cell_id': native_indices,
        'FlowElem_xcc': flow_elem_xcc,
        'FlowElem_ycc': flow_elem_ycc,
        'TWL': twl_values,
        'simulation_id': simulation_id
    })

    ds.close()
    return results_df


def process_all_simulations(input_dir, output_dir):
    """
    Process all simulation files and create ML-ready datasets.

    Parameters:
        input_dir (str): Directory containing all .nc files
        output_dir (str): Directory to save processed data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process first file to get reference grid
    nc_files = sorted(Path(input_dir).glob('*.nc'))
    if not nc_files:
        raise ValueError(f"No .nc files found in {input_dir}")

    print("Processing first file to establish reference grid...")
    reference_df = extract_twl_data(nc_files[0], 0)
    reference_coords = reference_df[['grid_cell_id', 'FlowElem_xcc', 'FlowElem_ycc']]

    # Save coordinate reference with additional mesh information
    save_grid_reference(nc_files[0], reference_coords, f"{output_dir}/grid_reference.csv")
    print(f"Saved grid reference with {len(reference_coords)} cells")

    # Process all files
    all_twls = []
    print(f"\nProcessing {len(nc_files)} simulation files...")

    for i, file_path in enumerate(nc_files):
        print(f"Processing simulation {i + 1}/{len(nc_files)}: {file_path.name}")

        try:
            df = extract_twl_data(file_path, i)

            # Verify grid consistency
            if not validate_grid_consistency(df, reference_df):
                raise ValueError(f"Grid mismatch in file {file_path.name}")

            # Store TWLs with simulation ID
            all_twls.append(df[['grid_cell_id', 'TWL', 'simulation_id']])

        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            continue

    # Combine all TWLs
    print("\nCombining all simulation results...")
    combined_twls = pd.concat(all_twls, ignore_index=True)

    # Save in multiple formats for ML processing
    save_ml_ready_data(combined_twls, output_dir)

    return reference_coords, combined_twls


def validate_grid_consistency(df1, df2):
    """
    Validate that two datasets maintain the same grid structure.

    Returns:
        bool: True if grids match, raises ValueError if they don't
    """
    coord_match = np.allclose(df1['FlowElem_xcc'], df2['FlowElem_xcc']) and \
                  np.allclose(df1['FlowElem_ycc'], df2['FlowElem_ycc'])

    if not coord_match:
        raise ValueError("Grid coordinates don't match between files!")

    index_match = np.array_equal(df1['grid_cell_id'], df2['grid_cell_id'])

    if not index_match:
        raise ValueError("Grid cell indices don't match between files!")

    return True


def save_grid_reference(file_path, reference_coords, output_path):
    """
    Save the grid reference file with additional mesh information.

    Parameters:
        file_path (str): Path to the source .nc file
        reference_coords (pandas.DataFrame): Basic coordinate reference
        output_path (str): Where to save the enhanced reference file
    """
    ds = xr.open_dataset(file_path)

    # Create enhanced grid reference
    grid_ref = reference_coords.copy()

    # Add useful mesh information
    grid_ref['x_km'] = grid_ref['FlowElem_xcc'] / 1000  # Convert to km
    grid_ref['y_km'] = grid_ref['FlowElem_ycc'] / 1000
    grid_ref['cell_area'] = ds['FlowElem_bac'].values  # Cell areas
    grid_ref['bed_level'] = ds['FlowElem_bl'].values  # Bed levels

    ds.close()

    grid_ref.to_csv(output_path, index=False)
    print(f"Enhanced grid reference saved to {output_path}")


def save_ml_ready_data(combined_twls, output_dir):
    """
    Save the TWL data in multiple formats suitable for ML processing.

    Parameters:
        combined_twls (pandas.DataFrame): Combined TWL data from all simulations
        output_dir (str): Directory to save the processed data
    """
    # 1. Long format (good for data analysis and some ML frameworks)
    combined_twls.to_csv(f"{output_dir}/all_twls_long.csv", index=False)

    # 2. Wide format (each row is a grid cell, columns are simulations)
    twls_wide = combined_twls.pivot(
        index='grid_cell_id',
        columns='simulation_id',
        values='TWL'
    ).reset_index()
    twls_wide.columns = ['grid_cell_id'] + [f'sim_{i}' for i in range(len(twls_wide.columns) - 1)]
    twls_wide.to_csv(f"{output_dir}/all_twls_wide.csv", index=False)

    # 3. NumPy arrays (ready for deep learning frameworks)
    twls_array = twls_wide.iloc[:, 1:].values  # Exclude grid_cell_id
    np.save(f"{output_dir}/twls_array.npy", twls_array)

    print("\nSaved data in multiple formats:")
    print(f"1. Long format: {output_dir}/all_twls_long.csv")
    print(f"2. Wide format: {output_dir}/all_twls_wide.csv")
    print(f"3. NumPy array: {output_dir}/twls_array.npy")

    # Print shapes for verification
    print("\nData shapes:")
    print(f"Long format: {combined_twls.shape}")
    print(f"Wide format: {twls_wide.shape}")
    print(f"NumPy array: {twls_array.shape}")


if __name__ == "__main__":
    # File paths - replace with actual paths
    input_dir = "Z:\School\Capstone\Data\Result_map\sim_1"
    output_dir = "processed_data"

    try:
        reference_coords, combined_twls = process_all_simulations(input_dir, output_dir)
        print("\nProcessing complete!")

        # Print some summary statistics
        print("\nSummary Statistics:")
        print(f"Total grid cells: {len(reference_coords)}")
        print(f"Total simulations processed: {len(combined_twls['simulation_id'].unique())}")
        print(f"\nTWL value ranges:")
        print(f"Min TWL: {combined_twls['TWL'].min():.3f}")
        print(f"Max TWL: {combined_twls['TWL'].max():.3f}")
        print(f"Mean TWL: {combined_twls['TWL'].mean():.3f}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")