import logging
import torch
import numpy as np
from src.evaluation.metrics import calculate_metrics, update_metrics, log_metrics
from alive_progress import alive_bar


def persistence(dataset, num_predictions=3):
    
    input_sequences = dataset.tensors[0].numpy()[:, :-3, :5]
    ground_truths = dataset.tensors[1].numpy()
    
    last_values = input_sequences[:, -1, :] 
    predictions = np.repeat(last_values[:, np.newaxis, :], 3, axis=1)  

    metrics = {'normalized': {'MAE': [], 'MSE': [], 
                              'RMSE': [], 'R2': [], 
                              'MAPE': [], 'MRE': [], 
                              'MBE': [], 'MAAPE': [], 
                              'SDE': []}}

    num_series = 5  
    with alive_bar(input_sequences.shape[0], bar="smooth", spinner="waves", length=85) as persistence_bar:

        for i in range(input_sequences.shape[0]):  
            for z in range(num_series):
                                
                pred = predictions[i, :, z][np.newaxis, :]
                truth = ground_truths[i, :, z][np.newaxis, :]
                
                normalized_metrics = calculate_metrics(torch.from_numpy(truth), torch.from_numpy(pred))
                update_metrics(metrics['normalized'], **normalized_metrics)
            persistence_bar()
                
    logging.info("Testing process completed.")
    overall_means = log_metrics(metrics)
    
    return predictions
