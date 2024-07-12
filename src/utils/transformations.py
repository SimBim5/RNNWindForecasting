import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def normalise(data, scaler_type, min_max):
    strategy = normalization_strategy(scaler_type)
    normalized_data, scalers = strategy(data, min_max)
    return normalized_data, scalers

def normalization_strategy(scaler_type):
    if scaler_type == "robust_minmax_normalize":
        return robust_minmax_normalize
    elif scaler_type == "max_value_normalize":
        return max_value_normalize
    else:
        raise ValueError("Unsupported scaler type")

def robust_minmax_normalize(data, min_max=None):
    if isinstance(data, pd.DataFrame):
        normalized_data = pd.DataFrame(index=data.index, columns=data.columns)
        scalers = {}
        for col in data.columns:
            robust_scaler = RobustScaler()
            robust_scaled_data = robust_scaler.fit_transform(data[col].values.reshape(-1, 1))

            minmax_scaler = MinMaxScaler()
            final_scaled_data = minmax_scaler.fit_transform(robust_scaled_data).flatten()

            normalized_data[col] = final_scaled_data
            scalers[col] = (robust_scaler, minmax_scaler)
        return normalized_data, scalers
    elif isinstance(data, np.ndarray):        
        scalers = {}
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        robust_scaler = RobustScaler()
        robust_scaled_data = robust_scaler.fit_transform(data)

        minmax_scaler = MinMaxScaler()
        final_scaled_data = minmax_scaler.fit_transform(robust_scaled_data)
        
        if final_scaled_data.shape[1] == 1:
            final_scaled_data = final_scaled_data.flatten()
        
        scalers['all_data'] = (robust_scaler, minmax_scaler)
        return final_scaled_data, scalers
    else:
        raise TypeError("Data must be a pandas DataFrame or a numpy ndarray.")

def max_value_normalize(data, min_max):
    if isinstance(data, pd.DataFrame):
        normalized_data = pd.DataFrame(index=data.index, columns=data.columns)
        scalers = {}
        for col in data.columns:
            max_val = min_max[col][0]
            normalized_col = data[col] / max_val
            normalized_data[col] = normalized_col
            scalers[col] = {'max': max_val}            
        return normalized_data, scalers
    elif isinstance(data, np.ndarray):
        scalers = {}
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            is_flattened = True
        else:
            is_flattened = False
        normalized_data = np.zeros(data.shape)
        for i in range(data.shape[1]):
            max_val = min_max[col][0]
            normalized_data[:, i] = data[:, i] / max_val
            scalers[i] = {'max': max_val}
        if is_flattened:
            normalized_data = normalized_data.flatten()
        return normalized_data, scalers
    else:
        raise TypeError("Data must be a pandas DataFrame or a numpy ndarray.")

def inverse_normalise(data, scalers, scaler_type):
    strategy = inverse_normalization_strategy(scaler_type)
    inverse_data = strategy(data, scalers)
    return inverse_data

def inverse_normalization_strategy(scaler_type):
    if scaler_type == "robust_minmax_normalize":
        return inverse_robust_minmax_normalize
    elif scaler_type == "max_value_normalize":
        return inverse_max_value_normalize
    else:
        raise ValueError("Unsupported scaler type")

def inverse_robust_minmax_normalize(data, scalers):
    inverse_data = np.zeros_like(data)
    for i, key in enumerate(scalers.keys()):
        robust_scaler, minmax_scaler = scalers[key]
        data_column = data[:, :, i].reshape(-1, 1)
        data_column = minmax_scaler.inverse_transform(data_column)
        data_column = robust_scaler.inverse_transform(data_column)
        inverse_data[:, :, i] = data_column.reshape(data.shape[0], data.shape[1])
    return inverse_data

def inverse_max_value_normalize(data, scalers):
    inverse_data = np.zeros_like(data)
    for i, key in enumerate(scalers.keys()):
            max_value = scalers[key]
            data_column = data[:, :, i]
            inverse_data[:, :, i] = data_column * max_value
    return inverse_data