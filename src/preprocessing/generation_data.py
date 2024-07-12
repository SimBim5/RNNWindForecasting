import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
from alive_progress import alive_bar
from src.preprocessing.utils import fill_data, min_max_dataframe
from src.utils.transformations import normalise
from src.clustering.clustering import cluster

def load_wind_data(data_path):
    file_extension = data_path.split('.')[-1]
    if file_extension == 'xlsx':
        data_full = pd.read_excel(data_path)
    elif file_extension == 'parquet':
        data_full = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format: " + file_extension)

    if 'SETTLEMENTDATE' in data_full.columns:
        timestamps = data_full[['SETTLEMENTDATE']]
        data = data_full.drop(columns=['SETTLEMENTDATE'])
    return data, timestamps

def preprocess_wind_data(data_path):
    logging.info(f"Loading data from: {data_path}")
    data, timestamps = load_wind_data(data_path)    
    logging.info("Data loaded. Checking for missing values...")
    columns_before = data.shape[1]
    data = data.dropna(axis=1, how='all')
    columns_after = data.shape[1]    
    dropped_nan_columns = columns_before - columns_after
    logging.info(f"Dropped {dropped_nan_columns} NaN columns.")
    original_nan_count = data.isna().sum().sum()
    data = fill_data(data)    
    filled_nan_count = original_nan_count - data.isna().sum().sum()
    logging.info(f"Filled {filled_nan_count} missing values.")
    columns_before = data.shape[1]
    mask = (data.notna() & data.ne(0)).any(axis=0)
    data = data.loc[:, mask]
    columns_after = data.shape[1]
    logging.info(f"Dropped {columns_before - columns_after} columns with only zeros.")
    return data, timestamps
    
def cancel_zeros(complete_data_farms, data_block, min_max, zero_threshold_percentage, consecutive_zeros_limit):    
    def has_consecutive_below_threshold(series, threshold):
        zero_count = 0
        for value in series:
            if value < threshold:
                zero_count += 1
                if zero_count >= consecutive_zeros_limit:
                    return True
            else:
                zero_count = 0
        return False

    valid_farms = []
    for farm in complete_data_farms:
        threshold = min_max[farm].iloc[0] * zero_threshold_percentage
        if not has_consecutive_below_threshold(data_block[farm], threshold):
            valid_farms.append(farm)
    return valid_farms

def interpolate_zeros(data_block, zero_threshold):
    data_block = data_block.mask(data_block < zero_threshold, np.nan)
    return data_block.interpolate(method='linear', limit_area='inside').bfill().ffill()

def create_dataset(data, sequence_length, prediction_length, timestamps, mode, num_series, scaler_type, cluster_type, max_wind):    
    min_max = min_max_dataframe(data)
    input_sequences, ground_truth_values = [], []
    timestamp_blocks, farm_names_blocks, scalers_blocks, sorted_farms = [], [], [], []
    total_length = sequence_length + prediction_length
    
    if mode == "predict_all":
        shift_step = 1
        data = data.iloc[203114:]
        timestamps = timestamps.iloc[203114:]
    else:
        shift_step = total_length
    
    prev_complete_data_farms = set()
    
    with alive_bar(math.floor(len(data) // shift_step), bar="smooth", spinner="waves", length=85) as create_bar:
        for block_start in range(0, len(data) - total_length + 1, shift_step):
            data_block = data.iloc[block_start:block_start + total_length]
            
            timestamp_block = timestamps.iloc[block_start:block_start + total_length]

            non_nan_columns = [col for col in data_block.iloc[:total_length].columns 
                            if not data_block.iloc[:total_length][col].isna().any()]
            
            complete_data_farms = cancel_zeros(non_nan_columns, data_block, min_max, 
                                zero_threshold_percentage=0.1, consecutive_zeros_limit=5)
                                    
            if len(complete_data_farms) < num_series:
                create_bar()  
                continue

            data_block = interpolate_zeros(data_block, zero_threshold=0.01)
            if set(complete_data_farms) != prev_complete_data_farms:
                sorted_farms = cluster(complete_data_farms, num_series, cluster_type, min_max)
                prev_complete_data_farms = set(complete_data_farms)

            data_block = data_block.reindex(sorted_farms, axis=1)
                        
            for i in range(0, data_block.shape[1], num_series):
                selected_columns = data_block.columns[i:i + num_series]
                if len(selected_columns) != num_series:
                    create_bar()  
                    continue
                subset_data, subset_scalers = normalise(data_block[selected_columns], scaler_type, min_max)
                future_weather_zeros = np.zeros((prediction_length, subset_data.shape[1]))
                input_sequences.append(np.vstack((subset_data.iloc[:sequence_length].values, future_weather_zeros)))
                ground_truth_values.append(subset_data.iloc[sequence_length:total_length].values)
                timestamp_blocks.append(timestamp_block.iloc[:total_length].values)
                farm_names_blocks.append(selected_columns)
                scalers_blocks.append(subset_scalers)

                # Optional: plotting of input data can be done here
                # plot_input_data(input_seq, ground_truth, sequence_length, prediction_length)
                
            create_bar()
    return np.array(input_sequences), np.array(ground_truth_values), np.array(timestamp_blocks), np.array(farm_names_blocks), np.array(scalers_blocks), scaler_type