import os
import pandas as pd
import logging
import numpy as np

from alive_progress import alive_bar
from src.utils.transformations import normalise

def get_weather_data(config, timestamp, farm_names, prediction_length, norm):
    weather_path = "data/weather_data/weather/parquet"
    logging.info(f"Loading weather data from: {weather_path}")
    weather_data_dict = load_weather_data(config)
    logging.info(f"Weather Data Loaded. Creating Dataset ...")
    weather_data = filter_data(weather_data_dict, timestamp, farm_names)
        
    logging.info(f"Dataset Created. Normalising Weather Data and Adding Forecasting Error ...")
    normalized_weather_data = np.zeros_like(weather_data)
    scalers_weather = {}
    with alive_bar(weather_data.shape[0] * weather_data.shape[2] * weather_data.shape[3], bar="smooth", spinner="waves", length=85) as forecast_bar:
        for i in range(weather_data.shape[0]):
            for j in range(weather_data.shape[2]):
                for k in range(weather_data.shape[3]):
                    data_array = weather_data[i, :, j, k]
                    data_df = pd.DataFrame(data_array)
                    normalized_data, data_scalers = normalise(data_df, norm, min_max=None)
                    normalized_data_values = normalized_data.values.flatten()
                    normalized_data_values[-3:] = add_forecast_error(normalized_data_values[-3:])
                    normalized_weather_data[i, :, j, k] = normalized_data_values
                    scalers_weather[(i, j, k)] = data_scalers
                    forecast_bar()
    logging.info("Normalised and Forecasting Error Added. Converting to tensor")
    return normalized_weather_data, scalers_weather

def load_weather_data(config):
    weather_config = config["weather"]
    weather_data_dict = {}

    with alive_bar(len(os.listdir("data/weather_data/weather_interpolated/parquet")), bar="smooth", spinner="waves", length=85) as load_bar:
        for farm_name in os.listdir("data/weather_data/weather_interpolated/parquet"):
            full_df = pd.read_parquet(f'data/weather_data/weather_interpolated/parquet/{farm_name}')
            filtered_columns = [column for column in [feature for feature, enabled in weather_config.items() if enabled] if column in full_df.columns]                
            filtered_df = full_df[filtered_columns]
            weather_data_dict[os.path.splitext(farm_name)[0]] = filtered_df
            load_bar()

    return weather_data_dict

def filter_data(weather_data_dict, timestamps, farm_names):
    all_examples_data = []  
    timestamps = np.squeeze(timestamps)
    first_entries = pd.to_datetime(timestamps[:, 0]).strftime('%Y-%m-%d %H:%M:%S')
    last_entries = pd.to_datetime(timestamps[:, -1]).strftime('%Y-%m-%d %H:%M:%S')
    
    with alive_bar(len(farm_names), bar="smooth", spinner="waves", length=85) as filter_bar:
        for i, (first_entry, last_entry) in enumerate(zip(first_entries, last_entries)):
            example_data = []
            for names in farm_names[i]:
                farm_data = weather_data_dict[names]
                mask = (farm_data.index >= first_entry) & (farm_data.index <= last_entry)
                filtered_farm_data = farm_data.loc[mask]            
                example_data.append(filtered_farm_data.to_numpy())
            filter_bar()

            example_array = np.array(example_data)
            all_examples_data.append(example_array)
    
    return np.array(all_examples_data).transpose(0, 2, 1, 3)

def add_forecast_error(data):
    systematic_error = 0.05 * data
    error_std_dev = 0.1 * np.mean(data)

    biased_data = data + systematic_error
    random_error = np.random.normal(0, error_std_dev, data.shape)
    simulated_data = biased_data + random_error
    return simulated_data