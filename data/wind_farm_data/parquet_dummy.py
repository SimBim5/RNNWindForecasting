import pandas as pd

def create_dummy_dataset(input_filepath, output_filepath, num_columns, num_rows):
    """
    Create a dummy dataset with a specified number of columns and rows.

    Args:
        input_filepath: Path to the input parquet file.
        output_filepath: Path to save the dummy parquet file.
        num_columns: Number of columns to keep in the dummy dataset.
        num_rows: Number of rows to keep in the dummy dataset.
    """
    # Load the dataset
    df = pd.read_parquet(input_filepath)

    # Select the first 'num_columns' columns
    if num_columns < len(df.columns):
        df = df.iloc[:, :num_columns]
    else:
        print("Warning: Requested more columns than available. Using all columns.")

    # Shorten the dataset to 'num_rows' rows
    if num_rows < len(df):
        df = df.iloc[:num_rows, :]
    else:
        print("Warning: Requested more rows than available. Using all rows.")

    # Save the smaller dataset
    df.to_parquet(output_filepath)

if __name__ == "__main__":
    INPUT_FILE = 'wind_data_train.parquet'
    OUTPUT_FILE = 'wind_data_dummy.parquet'
    NUMBER_OF_COLUMNS = 10  # Number of columns to keep
    NUMBER_OF_ROWS = 5000   # Number of rows to keep

    create_dummy_dataset(INPUT_FILE, OUTPUT_FILE, NUMBER_OF_COLUMNS, NUMBER_OF_ROWS)