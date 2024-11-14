import scipy.io
import numpy as np
from pathlib import Path


def read_matlab_file(filepath):
    """
    Read a MATLAB .mat file and display its structure and contents.

    Args:
        filepath (str): Path to the MATLAB file

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the file is not a valid MATLAB file
        Exception: For other unexpected errors
    """
    mat_contents = None
    try:
        if not Path(filepath).is_file():
            raise FileNotFoundError(f"File not found: {filepath}")

        mat_contents = scipy.io.loadmat(filepath)

        print(f"\nStructure of MATLAB file: {filepath}\n")
        print("=" * 50)

        for var_name, var_content in mat_contents.items():
            # Skip built-in MATLAB variables that start with '__'
            if not var_name.startswith('__'):
                print(f"\nVariable name: {var_name}")
                print(f"Type: {type(var_content)}")
                print(f"Shape: {var_content.shape}")

                # Print first few elements if it's an array
                if isinstance(var_content, np.ndarray):
                    # Handle different data types
                    if var_content.dtype.kind in ['U', 'S']:  # String or Unicode
                        print("Content type: String/Character array")
                        if var_content.size > 0:
                            print("First element(s):", var_content.flatten()[0])
                    else:  # Numeric arrays
                        print("Content type: Numeric array")
                        if var_content.size > 0:
                            print("First few elements:", var_content.flatten()[:3])

                print("-" * 30)

    except FileNotFoundError as e:
        print(f"Error: The file '{filepath}' was not found.")
        raise
    except ValueError as e:
        print(f"Error: '{filepath}' is not a valid MATLAB file or is corrupted.")
        raise
    except Exception as e:
        print(f"Unexpected error occurred while processing the file: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    file_path = "Z:\School\Capstone\Data\Design_points_MDA.mat"
    try:
        read_matlab_file(file_path)
    except Exception as e:
        print(f"Failed to process MATLAB file: {str(e)}")