if __name__ == "__main__":
    import pandas as pd

    print("Wide format first 5 rows:")
    wide_df = pd.read_csv("processed_data/all_twls_wide.csv", nrows=5)
    print(wide_df.columns)
    print(wide_df.head())

    print("\nLong format first 5 rows:")
    long_df = pd.read_csv("processed_data/all_twls_long.csv", nrows=5)
    print(long_df.columns)
    print(long_df.head())

    print("\nGrid reference first 5 rows:")
    grid_ref = pd.read_csv("processed_data/grid_reference.csv", nrows=5)
    print(grid_ref.columns)
    print(grid_ref.head())
