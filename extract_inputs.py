import scipy.io
import pandas as pd
from pathlib import Path


def extract_and_format_inputs(filepath, num_samples=750):
    """
    Extract and format input variables for the TWL model.

    Args:
        filepath (str): Path to the MATLAB .mat file
        num_samples (int): Number of samples to extract

    Returns:
        pandas.DataFrame: Formatted input data ready for the TWL model

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the file is not a valid MATLAB file
    """
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        mat_contents = scipy.io.loadmat(filepath)

        # Define expected variables based on their types
        variable_groups = {
            'tidal': ['pM2', 'pK1', 'pO1', 'pS2', 'pN2', 'pP1', 'pSA', 'pQ1', 'pK2', 'pSSA'],
            'waves': ['hs1', 'tp1', 'dir1', 'hs2', 'tp2', 'dir2', 'hs3', 'tp3', 'dir3'],
            'atmospheric': ['slp', 'wdu', 'wdv'],
            'river': ['qsac', 'qsan'],
            'sea_level': ['MMSLA']
        }

        data_dict = {}

        # Extract and flatten each variable
        for group, variables in variable_groups.items():
            for var in variables:
                if var in mat_contents:
                    data_dict[var] = mat_contents[var][:num_samples].flatten()
                else:
                    raise ValueError(f"Required variable {var} not found in MATLAB file")

        df = pd.DataFrame(data_dict)

        if len(df.columns) != 25:
            raise ValueError(f"Expected 25 input features, got {len(df.columns)}")

        return df

    except ValueError as e:
        raise ValueError(f"Error formatting data: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")


def save_formatted_data(df, output_path):
    """
    Save the formatted data to a CSV file.

    Args:
        df (pandas.DataFrame): Formatted input data
        output_path (str or Path): Path to save the CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    input_path = Path("Z:/School/Capstone/Data/Design_points_MDA.mat")
    output_path = Path("Z:/School/Capstone/Data/twl_model_inputs.csv")

    try:
        df = extract_and_format_inputs(input_path)

        print("\nInput Data Summary:")
        print("=" * 50)
        print(df.describe())

        save_formatted_data(df, output_path)

    except Exception as e:
        print(f"Error: {str(e)}")