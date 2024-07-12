import numpy as np
import pandas as pd
import os
from geopy.distance import great_circle
from src.preprocessing.utils import plot_correlation_matrix, plot_distance_matrix, safe_corrcoef, save_lagged_correlations_excel
from src.utils.transformations import normalise


def calculate_correlations(input_sequences, farm_names, config, norm):
    num_examples, time_series_length, num_series = input_sequences.shape
    num_pairs = num_series * (num_series - 1) // 2
    correlations = np.zeros((num_examples, num_pairs))
    normalised_correlations = np.zeros((num_examples, num_pairs))
    
    for i in range(num_examples):
        flattened_sequence = input_sequences[i].reshape(time_series_length, num_series)
        corr_matrix = safe_corrcoef(flattened_sequence)
        correlations[i, :] = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        normalised_correlations[i, :], _ = normalise(correlations[i, :], norm, min_max=None)

        if config.get('spatial', {}).get('plot', False):
            plot_correlation_matrix(corr_matrix, farm_names[i], input_sequences[i], i)
    
    return normalised_correlations

def calculate_distance_metrics(farm_names, config, norm):
    excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'weather_data', 'locations_wind_farms.xlsx')
    df = pd.read_excel(excel_path)
    
    num_examples, num_series = farm_names.shape
    num_pairs = num_series * (num_series - 1) // 2
    all_distances = np.zeros((num_examples, num_pairs))
    normalised_distance = np.zeros((num_examples, num_pairs))
    
    for example_index, farm_names_example in enumerate(farm_names):
        farm_names_list = farm_names_example.tolist()
        coordinates = df[df['Abbreviation'].isin(farm_names_list)].set_index('Abbreviation').reindex(farm_names_list)[['Latitude', 'Longitude']].values
        
        distances = np.zeros(num_pairs)
        index = 0
        for i in range(num_series):
            for j in range(i + 1, num_series):
                distances[index] = great_circle(coordinates[i], coordinates[j]).kilometers
                index += 1
        all_distances[example_index], _ = normalise(distances, norm, min_max=None)    
        if config.get('spatial', {}).get('plot', False):
            plot_distance_matrix(distances, farm_names[example_index], coordinates, example_index)
    return all_distances

def calculate_correlations_lagged(input_sequences, farm_names, config, max_lag):
    num_examples, sequence_length, num_farms = input_sequences.shape

    base_save_dir = 'results/spatial'
    os.makedirs(base_save_dir, exist_ok=True)
    lagged_correlations_per_example = {}
    
    for example_index in range(num_examples):
        lagged_correlations = {}
        best_lags = {}
        
        for i in range(num_farms):
            for j in range(num_farms):
                if i == j:
                    continue
                
                correlations_at_lags = np.zeros((max_lag+1,))

                for lag in range(max_lag + 1):
                    if lag > 0:
                        series_i_lag = input_sequences[example_index, :-lag, i]
                        series_j_lag = input_sequences[example_index, lag:, j]
                    else:
                        series_i_lag = input_sequences[example_index, :, i]
                        series_j_lag = input_sequences[example_index, :, j]

                    if len(series_i_lag) > 0 and len(series_j_lag) > 0:
                        correlation = np.corrcoef(series_i_lag, series_j_lag)[0, 1]
                        correlations_at_lags[lag] = correlation
                
                best_lag = np.argmax(correlations_at_lags)
                best_correlation = correlations_at_lags[best_lag]

                farm_pair_name = f"{farm_names[example_index, i]}-{farm_names[example_index, j]}"
                lagged_correlations[farm_pair_name] = correlations_at_lags
                best_lags[farm_pair_name] = (best_lag, best_correlation)
                
        lagged_correlations_per_example[example_index] = lagged_correlations
        if config.get('spatial', {}).get('plot', False):
            save_lagged_correlations_excel(base_save_dir, example_index, lagged_correlations, max_lag, best_lags)
    return lagged_correlations_per_example