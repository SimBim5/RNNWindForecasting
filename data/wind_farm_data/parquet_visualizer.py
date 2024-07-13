import pandas as pd
import matplotlib.pyplot as plt

def load_and_filter_data(filepath, start_date, end_date):
    """
    Load parquet data, filter based on date range, and ensure correct datetime format.

    Args:
        filepath: Path to the parquet file.
        start_date: Start date for filtering the data.
        end_date: End date for filtering the data.

    Returns:
        Filtered DataFrame.
    """
    df = pd.read_parquet(filepath)
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    return df[(df['SETTLEMENTDATE'] >= start_date) & (df['SETTLEMENTDATE'] <= end_date)]

def plot_data(df, title_suffix):
    """
    Plot each column in the DataFrame against the settlement date.

    Args:
        df: DataFrame to be plotted.
        title_suffix: Suffix for the plot title to distinguish between datasets.
    """
    for column in df.columns:
        if column != 'SETTLEMENTDATE':
            plt.figure(figsize=(10, 4))
            plt.plot(df['SETTLEMENTDATE'], df[column], label=column)
            plt.title(f'Energy Output for {column} - {title_suffix}')
            plt.xlabel('Settlement Date')
            plt.ylabel('Energy Output')
            plt.legend()
            plt.show()

def visualize_datasets(train_filepath, test_filepath, start_date, end_date):
    """
    Visualize both training and testing datasets.

    Args:
        train_filepath: Path to the training dataset parquet file.
        test_filepath: Path to the testing dataset parquet file.
        start_date: Start date for filtering the data.
        end_date: End date for filtering the data.
    """
    train_df = load_and_filter_data(train_filepath, start_date, end_date)
    test_df = load_and_filter_data(test_filepath, start_date, end_date)

    plot_data(train_df, "Training Set")
    plot_data(test_df, "Testing Set")

if __name__ == "__main__":
    TRAIN_FILE = 'wind_data_train.parquet'
    TEST_FILE = 'wind_data_test.parquet'
    START_DATE = '2010-01-01'
    END_DATE = '2023-12-31'
    visualize_datasets(TRAIN_FILE, TEST_FILE, START_DATE, END_DATE)