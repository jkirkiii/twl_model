import xarray as xr
import numpy as np
import pandas as pd


def extract_twl_data(file_path):
    """
    Extract Total Water Levels (TWL) from the specified NetCDF file.
    Takes the second value of 's1' at each grid cell and processes according to specifications.

    Parameters:
        file_path (str): Path to the SFBD_map.nc file

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
        'FlowElem_xcc': flow_elem_xcc,
        'FlowElem_ycc': flow_elem_ycc,
        'TWL': twl_values
    })

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
    print("\nSample of the data:")
    print(df.head())


def save_results(df, output_path):
    """
    Save the results to a file.

    Parameters:
        df (pandas.DataFrame): The data to save
        output_path (str): Path where to save the results
    """
    # Save as CSV
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # File path - replace with actual path
    file_path = "Z:\School\Capstone\Data\Result_map\sim_1\SFBD_map.nc"
    output_path = "twl_results.csv"

    try:
        # Extract the data
        print("Extracting TWL data...")
        results = extract_twl_data(file_path)

        # Validate the results
        print("\nValidating extracted data:")
        validate_data(results)

        # Save the results
        save_results(results, output_path)

    except Exception as e:
        print(f"Error processing file: {str(e)}")