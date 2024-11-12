import xarray as xr
import numpy as np
import pandas as pd


def extract_twl_data(file_path, sort_method='yx'):
    """
    Extract Total Water Levels (TWL) from the specified NetCDF file.
    Takes the second value of 's1' at each grid cell and processes according to specifications.

    Parameters:
        file_path (str): Path to the SFBD_map.nc file
        sort_method (str): Sorting method to use ('yx', 'xy', or 'index')
            - 'yx': Sort by y-coordinate first, then x-coordinate (north-to-south, west-to-east)
            - 'xy': Sort by x-coordinate first, then y-coordinate (west-to-east, north-to-south)
            - 'index': Sort by original grid cell index if available

    Returns:
        pandas.DataFrame: DataFrame containing grid coordinates and processed TWL values
    """
    # Open the dataset
    ds = xr.open_dataset(file_path)

    # Extract the grid coordinates
    flow_elem_xcc = ds['FlowElem_xcc'].values
    flow_elem_ycc = ds['FlowElem_ycc'].values

    # Extract the second value of 's1' for each grid cell
    # The second value corresponds to the 72nd hour/final step
    twl_values = ds['s1'].isel(time=1).values  # Using index 1 for the second value

    # Set any negative or very small values to 0
    # Using a small epsilon to catch values very close to zero (~1e-5)
    twl_values = np.where(twl_values <= 1e-5, 0, twl_values)

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'grid_cell_index': np.arange(len(flow_elem_xcc)),  # Add original index
        'FlowElem_xcc': flow_elem_xcc,
        'FlowElem_ycc': flow_elem_ycc,
        'TWL': twl_values
    })

    # Sort the results based on the specified method
    if sort_method == 'yx':
        # Sort north-to-south, then west-to-east
        results_df = results_df.sort_values(
            by=['FlowElem_ycc', 'FlowElem_xcc'],
            ascending=[False, True]
        )
    elif sort_method == 'xy':
        # Sort west-to-east, then north-to-south
        results_df = results_df.sort_values(
            by=['FlowElem_xcc', 'FlowElem_ycc'],
            ascending=[True, False]
        )
    elif sort_method == 'index':
        # Sort by original grid cell index
        results_df = results_df.sort_values(by='grid_cell_index')

    # Close the dataset
    ds.close()

    return results_df


def validate_data(df):
    """
    Validate the extracted data to ensure it meets expectations.

    Parameters:
        df (pandas.DataFrame): The extracted data
    """
    print(f"Total number of grid cells: {len(df)}")
    print(f"Number of cells with TWL = 0: {(df['TWL'] == 0).sum()}")
    print(f"TWL range: {df['TWL'].min():.6f} to {df['TWL'].max():.6f}")
    print("\nCoordinate ranges:")
    print(f"X: {df['FlowElem_xcc'].min():.1f} to {df['FlowElem_xcc'].max():.1f}")
    print(f"Y: {df['FlowElem_ycc'].min():.1f} to {df['FlowElem_ycc'].max():.1f}")
    print("\nSample of the sorted data:")
    print(df.head())


def verify_sorting_consistency(df1, df2):
    """
    Verify that two processed files have the same spatial ordering.

    Parameters:
        df1, df2 (pandas.DataFrame): Two processed datasets to compare

    Returns:
        bool: True if sorting is consistent, False otherwise
    """
    coord_match = (
            df1['FlowElem_xcc'].equals(df2['FlowElem_xcc']) and
            df1['FlowElem_ycc'].equals(df2['FlowElem_ycc'])
    )
    return coord_match


def save_results(df, output_path):
    """
    Save the results to a file.

    Parameters:
        df (pandas.DataFrame): The data to save
        output_path (str): Path where to save the results
    """
    # Save as CSV with consistent float formatting
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # File path - replace with actual path
    file_path = "Z:\School\Capstone\Data\Result_map\sim_1\SFBD_map.nc"
    output_path = "twl_results.csv"
    sort_method = 'yx'  # Change this to 'xy' or 'index' if needed

    try:
        # Extract the data
        print(f"Extracting TWL data using {sort_method} sorting...")
        results = extract_twl_data(file_path, sort_method=sort_method)

        # Validate the results
        print("\nValidating extracted data:")
        validate_data(results)

        # Save the results
        save_results(results, output_path)

    except Exception as e:
        print(f"Error processing file: {str(e)}")