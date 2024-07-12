import torch
import logging 
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

def mean_absolute_error(ground_truths, predictions):
    absolute_differences = torch.abs(ground_truths - predictions)
    mae = torch.mean(absolute_differences)
    return mae

def mean_squared_error(ground_truth, predictions):
    squared_errors = (ground_truth - predictions) ** 2
    mse = torch.mean(squared_errors)
    return mse

def root_mean_squared_error(ground_truth, predictions):
    squared_errors = (ground_truth - predictions) ** 2
    mse = torch.mean(squared_errors)
    rmse = torch.sqrt(mse)
    return rmse

def r_squared(ground_truth, predictions): 
    ground_truth = ground_truth.numpy()
    predictions = predictions.numpy()
    x_mean = np.mean(ground_truth)
    y_mean = np.mean(predictions)
    
    diff_x = ground_truth - x_mean
    diff_y = predictions - y_mean
    
    denominator = np.sqrt(np.sum(diff_x**2)) * np.sqrt(np.sum(diff_y**2))
    
    correlation = np.sum(diff_x * diff_y) / denominator
    r_squared = correlation**2
    return r_squared

def mean_absolute_percentage_error(ground_truth, predictions):
    ground_truth, predictions = ground_truth.numpy(), predictions.numpy()  # Convert to numpy for division
    nonzero_indices = ground_truth != 0
    return np.mean(np.abs((ground_truth[nonzero_indices] - predictions[nonzero_indices]) / ground_truth[nonzero_indices])) * 100


def mean_relative_error(ground_truth, predictions):
    ground_truth, predictions = ground_truth.numpy(), predictions.numpy()
    nonzero_indices = ground_truth != 0
    return np.mean(np.abs(ground_truth[nonzero_indices] - predictions[nonzero_indices]) / ground_truth[nonzero_indices])

def mean_bias_error(ground_truth, predictions):
    ground_truth, predictions = ground_truth.numpy(), predictions.numpy()
    return np.mean(predictions - ground_truth)

def mean_arctangent_absolute_percentage_error(ground_truth, predictions):
    ground_truth, predictions = ground_truth.numpy(), predictions.numpy()
    nonzero_indices = ground_truth != 0
    return np.mean(np.arctan(np.abs((ground_truth[nonzero_indices] - predictions[nonzero_indices]) / ground_truth[nonzero_indices])))

def standard_deviation_of_errors(ground_truth, predictions):
    ground_truth, predictions = ground_truth.numpy(), predictions.numpy()
    return np.std(predictions - ground_truth)

def calculate_metrics(ground_truth, predictions):
    metrics_dict = {}
    metrics_dict['mae'] = mean_absolute_error(ground_truth, predictions)
    metrics_dict['mse'] = mean_squared_error(ground_truth, predictions)
    metrics_dict['rmse'] = root_mean_squared_error(ground_truth, predictions)
    metrics_dict['r2'] = r_squared(ground_truth, predictions)
    metrics_dict['mape'] = mean_absolute_percentage_error(ground_truth, predictions)
    metrics_dict['mre'] = mean_relative_error(ground_truth, predictions)
    metrics_dict['mbe'] = mean_bias_error(ground_truth, predictions)
    metrics_dict['maape'] = mean_arctangent_absolute_percentage_error(ground_truth, predictions)
    metrics_dict['sde'] = standard_deviation_of_errors(ground_truth, predictions)
    return metrics_dict

def update_metrics(metrics, mae, mse, rmse, r2, mape, mre, mbe, maape, sde):
    metrics['MAE'].append(mae)
    metrics['MSE'].append(mse)
    metrics['RMSE'].append(rmse)
    metrics['R2'].append(r2)
    metrics['MAPE'].append(mape)
    metrics['MRE'].append(mre)
    metrics['MBE'].append(mbe)
    metrics['MAAPE'].append(maape)
    metrics['SDE'].append(sde)
    
def log_metrics(metrics):
    overall_means = {}
    for metric_type in ['normalized', 'inv_normalized']:
        for name, values in metrics[metric_type].items():
            overall_mean = torch.mean(torch.tensor(values)).item()
            logging.info(f"{metric_type} {name}: {overall_mean:.4f}")
            overall_means[f"{metric_type} {name}"] = overall_mean
    return overall_means