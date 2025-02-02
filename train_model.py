import numpy as np
import torch
import pandas as pd
from pathlib import Path
from models.twl_model import TWLModel
from cross_validation import cross_validate_model
from evaluation.plotting import plot_cv_results


def load_model_inputs(input_file='processed_data/twl_model_inputs.csv'):
    """
    Load and validate model input features from CSV.

    Expected columns include tidal constituents (pM2, pK1, etc.),
    wave characteristics (hs1-3, tp1-3, dir1-3), atmospheric (slp, wdu, wdv),
    and river flows (qsac, qsan).
    """

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} not found")

    expected_columns = [
        'pM2', 'pK1', 'pO1', 'pS2', 'pN2', 'pP1', 'pSA', 'pQ1', 'pK2', 'pSSA',
        'MMSLA',
        'hs1', 'tp1', 'dir1',
        'hs2', 'tp2', 'dir2',
        'hs3', 'tp3', 'dir3',
        'slp', 'wdu', 'wdv',
        'qsac', 'qsan'
    ]

    if len(df) != 750:
        raise ValueError(f"Expected 750 rows, but got {len(df)}")

    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    X = df[expected_columns].values

    if X.shape != (750, 25):
        raise ValueError(f"Expected shape (750, 25), but got {X.shape}")

    return X


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    input_file = script_dir / 'processed_data/twl_model_inputs.csv'
    twl_file = script_dir / 'processed_data/twls_array.npy'
    log_file = 'train_log.txt'

    try:
        # Load input features
        X = load_model_inputs(input_file)
        with open(log_file, 'a') as f:
            f.write(f"Loaded input data with shape: {X.shape}\n")

        # Load target values (TWL data)
        twl_data = np.load(twl_file)
        twl_data = twl_data.T
        with open(log_file, 'a') as f:
            f.write(f"Loaded TWL data with shape: {twl_data.shape}\n")

        model_params = {
            'input_size': 25,
            'hidden_sizes': [256, 256, 256],
            'output_size': 179269,
            'dropout_rate': 0.3
        }

        # Perform cross-validation
        fold_results, cv_stats = cross_validate_model(
            X=X,
            y=twl_data,
            model_class=TWLModel,
            n_splits=5,
            **model_params
        )

        plot_cv_results(fold_results, log_file)

        # Select best model from cross-validation
        best_fold = min(fold_results, key=lambda x: x['metrics']['final_val_loss'])
        best_model = best_fold['model']
        best_scaler = best_fold['scaler']

        # Save best model and scaler
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'model_params': model_params,
            'cv_stats': cv_stats,
            'scaler_state': best_scaler
        }, 'best_model.pt')

        with open(log_file, 'a') as f:
            f.write("\nBest model saved from fold "
                   f"{best_fold['metrics']['fold']}\n")
            f.write(f"Best validation loss: "
                   f"{best_fold['metrics']['final_val_loss']:.6f}\n")

    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"\nError during model training: {str(e)}\n")
        raise e
