import os
import torch
import logging

from src.preprocessing.utils import save_weather_as_tensor, save_wind_as_tensors, save_as_dataset, max_wind_data
from src.utils.file_operations import get_cache_path, get_weather_cache_path, get_spatial_cache_path
from src.preprocessing.weather_data import get_weather_data
from src.preprocessing.generation_data import preprocess_wind_data, create_dataset
from src.preprocessing.spatial_data import calculate_correlations, calculate_distance_metrics


def load_or_create_dataset(data_path, sequence_length, prediction_length, num_series, norm, mode, cluster_type, config):
    max_wind = max_wind_data(data_path)
    
    cache_path = get_cache_path(data_path, sequence_length, prediction_length, num_series, norm, mode, cluster_type)
    if os.path.exists(cache_path):
        input_sequences_tensor, ground_truth_tensor, timestamp, farm_names, scaler, scaler_type = load_wind_data(cache_path)
    else:
        input_sequences_tensor, ground_truth_tensor, timestamp, farm_names, scaler, scaler_type = generate_wind_data(data_path, cache_path, sequence_length, prediction_length, mode, num_series, norm, cluster_type, max_wind)

    cache_path_weather = get_weather_cache_path(data_path, sequence_length, prediction_length, num_series, norm, mode, cluster_type, config)
    if os.path.exists(cache_path_weather):
        weather_data_tensor, weather_scalers = load_weather_data(cache_path_weather)
    else:
        weather_data_tensor, weather_scalers = generate_weather_data(config, timestamp, farm_names, prediction_length, norm, cache_path_weather)    
    
    examples, seq_length, farms, weather_features = weather_data_tensor.shape
    weather_data_reshaped = weather_data_tensor.reshape(examples, seq_length, -1)
    combined_data = torch.cat((input_sequences_tensor, weather_data_reshaped), dim=-1)
    dataset = save_as_dataset(combined_data, ground_truth_tensor)

    cache_path_spatial = get_spatial_cache_path(mode, sequence_length, prediction_length, num_series, cluster_type)
    if os.path.exists(cache_path_spatial):
        correlation, distance = load_spatial_data(cache_path_spatial)
    else:
        correlation, distance = generate_spatial_data(input_sequences_tensor, prediction_length, farm_names, config, norm, cache_path_spatial)    

    spatial_data = torch.cat((correlation, distance), dim=1)

    return dataset, timestamp, farm_names, scaler, scaler_type, combined_data.shape[-1], spatial_data
    
def load_wind_data(cache_path):
    logging.info(f"Loading Wind Farm Generation Data from cache: {cache_path}\n")
    cache_data = torch.load(cache_path)
    input_sequences_tensor = cache_data['input_sequences_tensor']
    ground_truth_tensor = cache_data['ground_truth_tensor']
    timestamp = cache_data['timestamp']
    farm_names = cache_data['farm_names']
    scaler = cache_data['scaler']
    scaler_type = cache_data['scaler_type']
    return input_sequences_tensor, ground_truth_tensor, timestamp, farm_names, scaler, scaler_type

def generate_wind_data(data_path, cache_path, sequence_length, prediction_length, mode, num_series, norm, cluster_type, max_wind):
    logging.info("Wind Farm Generation Dataset Cache not found. Create New Dataset. Preprocessing Data ...")
    data, timestamps = preprocess_wind_data(data_path)
    logging.info("Data Preprocessed. Creating Dataset ...")
    input_sequences, ground_truth, timestamp, farm_names, scaler, scaler_type = create_dataset(data, sequence_length, prediction_length, timestamps, mode, num_series, norm, cluster_type, max_wind)        
    input_sequences_tensor, ground_truth_tensor = save_wind_as_tensors(input_sequences, ground_truth)
    logging.info(f"Dataset created and converted to tensors.")
    cache_data = {
        'input_sequences_tensor': input_sequences_tensor,
        'ground_truth_tensor': ground_truth_tensor,
        'timestamp': timestamp,
        'farm_names': farm_names,
        'scaler': scaler,
        'scaler_type': scaler_type}
    logging.info(f"Caching dataset to {cache_path}\n")
    torch.save(cache_data, cache_path)
    return input_sequences_tensor, ground_truth_tensor, timestamp, farm_names, scaler, scaler_type

def load_weather_data(cache_path_weather):
    logging.info(f"Loading Weather Data from Cache: {cache_path_weather}\n")
    cache_weather_data = torch.load(cache_path_weather)
    weather_data_tensor = cache_weather_data['weather_data_tensor']
    weather_scalers = cache_weather_data['weather_scalers']
    return weather_data_tensor, weather_scalers

def generate_weather_data(config, timestamp, farm_names, prediction_length, norm, cache_path_weather):
    logging.info("Weather Data Cache not found. Create New Dataset. Preprocessing Data ...")
    weather_data, weather_scalers = get_weather_data(config, timestamp, farm_names, prediction_length, norm)        
    weather_data_tensor = save_weather_as_tensor(weather_data)
    logging.info(f"Converted Dataset to tensors.")
    cache_weather_data = {
        'weather_data_tensor': weather_data_tensor,
        'weather_scalers': weather_scalers}
    logging.info(f"Caching dataset to {cache_path_weather}\n")
    torch.save(cache_weather_data, cache_path_weather)
    return weather_data_tensor, weather_scalers

def load_spatial_data(cache_path_spatial):
    logging.info(f"Loading Spatial Data from cache: {cache_path_spatial}\n")
    cache_path_spatial = torch.load(cache_path_spatial)
    correlation = cache_path_spatial['correlation']
    distance = cache_path_spatial['distance']
    return correlation, distance

def generate_spatial_data(input_sequences_tensor, prediction_length, farm_names, config, norm, cache_path_spatial):
    logging.info("Spatial Data Cache not found. Create New Dataset. Calculating Correlations and Distances ...")
    logging.info("Calculating Correlations ...")
    correlation = calculate_correlations(input_sequences_tensor[:, :-prediction_length, :], farm_names, config, norm)
    logging.info("Correlations Calculated. Calculating Distances...")
    distance = calculate_distance_metrics(farm_names, config, norm)
    
    logging.info("Distances Calculated. Converting Spatial Data to Tensors...")
    correlation_tensor = torch.tensor(correlation, dtype=torch.float32)
    distance_tensor = torch.tensor(distance, dtype=torch.float32)
    cache_spatial_data = {
        'correlation': correlation_tensor,
        'distance': distance_tensor}
    logging.info(f"Caching Spatial Data to {cache_path_spatial}\n")
    torch.save(cache_spatial_data, cache_path_spatial)
    return correlation_tensor, distance_tensor