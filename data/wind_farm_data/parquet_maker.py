import pandas as pd
import numpy as np
import logging
import os 
from scipy.spatial.distance import pdist, squareform
from alive_progress import alive_bar

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXCEL_FILES = [f"excel/aemo_{year}_wind.xlsx" for year in range(2010, 2024)]

def merge_excel_files(file_paths):
    """
    Merge multiple Excel files into a single DataFrame.
    """    
    merged_df = pd.DataFrame()
    with alive_bar(len(file_paths), bar="smooth", spinner="waves", length=85) as bar:
        for file in file_paths:
            try:
                df = pd.read_excel(file)
                df.columns = [col.upper() for col in df.columns]
                
                if 'SETTLEMENTDATE' in df.columns:
                    duplicates = df[df.duplicated(subset='SETTLEMENTDATE', keep=False)]
                    
                    if not duplicates.empty:
                        exact_duplicates = duplicates[duplicates.duplicated(keep=False)]
                        if not exact_duplicates.empty:
                            unique_dates_exact = exact_duplicates['SETTLEMENTDATE'].dt.strftime('%Y-%m-%d %H:%M:%S').unique()
                            for date in unique_dates_exact:
                                logging.info(f"Exact duplicate entries found for date {date} in file {file}.")
                        df.drop_duplicates(subset='SETTLEMENTDATE', keep='first', inplace=True)

                        diff_duplicates = duplicates.drop_duplicates(subset='SETTLEMENTDATE', keep=False)
                        if not diff_duplicates.empty:
                            unique_dates_diff = diff_duplicates['SETTLEMENTDATE'].dt.strftime('%Y-%m-%d %H:%M:%S').unique()
                            for date in unique_dates_diff:
                                logging.info(f"Different duplicate entries (to be reviewed) found for date {date} in file {file}.")
                
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                bar()  
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")
                bar()  
    return merged_df

def clean_dataframe(df):
    """
    Clean and prepare the DataFrame by filling NaNs and sorting.
    """
    df.fillna(pd.NA, inplace=True)
    try:
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
        df.sort_values(by='SETTLEMENTDATE', inplace=True)
    except KeyError as e:
        logging.error(f"SETTLEMENTDATE column not found: {e}")
    return df

def main():
    """
    Main function to orchestrate the data processing workflow.
    """
    logging.info("Excel files are getting merged into a single dataframe - this might take a while.")
    merged_df = merge_excel_files(EXCEL_FILES)
    logging.info("Excel files are merged.")
    logging.info("Clean and prepare the DataFrame by filling NaNs and sorting.")
    clean_df = clean_dataframe(merged_df)
    logging.info("DataFrame Cleaned.")

    for column in clean_df.columns:
        if column != 'SETTLEMENTDATE':
            clean_df[column] = clean_df[column].apply(lambda x: max(x, 0))

    logging.info("Split Wind Farms in Testing and Training Set.")
    split_index = int(len(clean_df) * 0.85)
    train_df = clean_df.iloc[:split_index]
    test_df = clean_df.iloc[split_index:]
    logging.info("Dataset is splitted.")

    train_df.to_parquet('wind_data_train.parquet')
    test_df.to_parquet('wind_data_test.parquet')

    print(train_df)
    print(test_df)
    print(test_df.shape)
    print(train_df.shape)

if __name__ == "__main__":
    main()