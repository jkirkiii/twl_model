import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


def extract_twl_data(file_path, simulation_id):
    """
    Extract Total Water Levels (TWL) from a single simulation file,
    preserving native grid cell indexing.

    Parameters:
        file_path (str): Path to the SFBD_map.nc file
        simulation_id (int): Identifier for this simulation run

    Returns:
        pandas.DataFrame: DataFrame containing grid coordinates and TWL values
    """
    ds = xr.open_dataset(file_path)

    # Print available variables to inspect grid structure
    print("Available variables in dataset:")
    for var_name, var in ds.variables.items():
        print(f"{var_name}: {var.dims}")

    # Extract coordinates
    flow_elem_xcc = ds['FlowElem_xcc'].values
    flow_elem_ycc = ds['FlowElem_ycc'].values

    # Look for native grid indexing
    # Common names in Delft3D FM might include:
    possible_index_vars = ['FlowElem', 'mesh2d_nFaces', 'nFaces', 'mesh2d_face_nodes']
    native_index = None

    for var in possible_index_vars:
        if var in ds.variables:
            print(f"\nFound potential grid index variable: {var}")
            print(f"Shape: {ds[var].shape}")
            print(f"Attributes: {ds[var].attrs}")
            native_index = ds[var].values
            break

    # Extract TWL values
    twl_values = ds['s1'].isel(time=1).values
    twl_values = np.where(twl_values <= 1e-5, 0, twl_values)

    # Create DataFrame
    results_df = pd.DataFrame({
        'FlowElem_xcc': flow_elem_xcc,
        'FlowElem_ycc': flow_elem_ycc,
        'TWL': twl_values,
        'simulation_id': simulation_id
    })

    # Add grid cell indexing
    if native_index is not None:
        results_df['grid_cell_id'] = native_index
        print("\nUsing native grid cell indexing from the file")
    else:
        print("\nWarning: No native grid cell indexing found.")
        print("Creating grid_cell_id based on sorted coordinate position.")
        print("Please verify if this is appropriate for your use case.")

        # Sort by coordinates and create index
        results_df = results_df.sort_values(
            by=['FlowElem_ycc', 'FlowElem_xcc'],
            ascending=[False, True]
        )
        results_df['grid_cell_id'] = np.arange(len(results_df))

    ds.close()
    return results_df


def inspect_grid_structure(file_path):
    """
    Inspect the grid structure of the NetCDF file to understand indexing.

    Parameters:
        file_path (str): Path to the SFBD_map.nc file
    """
    ds = xr.open_dataset(file_path)

    print("File Structure Analysis:")
    print("------------------------")
    print("\nDimensions:")
    for dim_name, dim in ds.dims.items():
        print(f"{dim_name}: {dim}")

    print("\nVariables:")
    for var_name, var in ds.variables.items():
        print(f"\n{var_name}:")
        print(f"  Shape: {var.shape}")
        print(f"  Dimensions: {var.dims}")
        if var.attrs:
            print("  Attributes:")
            for attr_name, attr_value in var.attrs.items():
                print(f"    {attr_name}: {attr_value}")

    ds.close()


if __name__ == "__main__":
    file_path = "Z:\School\Capstone\Data\Result_map\sim_1\SFBD_map.nc"

    try:
        # First, inspect the file structure
        print("Analyzing file structure...")
        inspect_grid_structure(file_path)

        # Then extract data
        print("\nExtracting data...")
        results = extract_twl_data(file_path, simulation_id=0)

        print("\nSample of extracted data:")
        print(results.head())

    except Exception as e:
        print(f"Error during processing: {str(e)}")