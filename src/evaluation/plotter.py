import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import torch

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
from bokeh.io import reset_output
from bokeh.models import DatetimeTickFormatter
from plotly.subplots import make_subplots
from src.evaluation.metrics import calculate_metrics
from src.utils.transformations import inverse_normalise
from src.utils.file_operations import save_to_file

def plot_results(input_sequences, ground_truths, means, timestamps, farm_names, save_dir, plot_index, scaler, scaler_type):
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1.5})

    input_sequences_np = input_sequences.numpy()  
    ground_truths_np = ground_truths.numpy()  
    means_np = means.numpy()  

    gt = inverse_normalise(ground_truths_np, scaler, scaler_type)
    pt = inverse_normalise(means_np, scaler, scaler_type)
        
    gt = torch.from_numpy(gt)
    pt = torch.from_numpy(pt)

    full_sequences_gt = np.concatenate((input_sequences_np, ground_truths_np), axis=1)
    full_sequences_pred = np.concatenate((input_sequences_np, means_np), axis=1)

    full_ground_truth = inverse_normalise(full_sequences_gt, scaler, scaler_type)
    full_predictions = inverse_normalise(full_sequences_pred, scaler, scaler_type)

    num_series = input_sequences.shape[2]

    palette = sns.color_palette("mako", 2*num_series)

    data_dict = {
        'input_sequences': input_sequences_np.tolist(),
        'ground_truths': ground_truths_np.tolist(),
        'means': means_np.tolist(),
        'timestamps': timestamps[0, :, 0].tolist(),
        'farm_names': farm_names[0],
    }

    plt.figure(figsize=(15, 8))
    timestamps_flattened = pd.to_datetime(timestamps[0, :, 0].flatten())
    date_str = timestamps_flattened[0].strftime("%Y-%m-%d") 

    for i in range(num_series):
        color_gt = palette[2*i]
        color_pred = palette[2*i + 1]

        series_gt = full_ground_truth[0, :, i]
        series_pred = full_predictions[0, :, i]

        ax = plt.subplot(num_series, 1, i + 1)
        sns.lineplot(x=timestamps_flattened, y=series_pred, label='Prediction', marker='x', color=color_pred, ax=ax, linewidth='2')
        sns.lineplot(x=timestamps_flattened, y=series_gt, label='Ground Truth', marker='o', color=color_gt, ax=ax, linewidth='2')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='grey')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle='--', linewidth='0.5', color='grey')
        ax.set_xticks(timestamps_flattened[::2])
        ax.set_xlabel("")
        
        sns.despine()  
        ax.legend(loc='center left')
        plt.tight_layout()
        
    save_to_file(data_dict, os.path.join(save_dir, f"{plot_index}.txt"))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(save_dir, f"{plot_index}.png"))  
    plt.close()

def plot_results_with_metrics(input_sequences, ground_truths, means, timestamps, farm_names, save_dir, plot_index, scaler, scaler_type):
    import seaborn as sns
    input_sequences_np = input_sequences.numpy()  
    ground_truths_np = ground_truths.numpy()  
    means_np = means.numpy()  

    gt = inverse_normalise(ground_truths_np, scaler, scaler_type)
    pt = inverse_normalise(means_np, scaler, scaler_type)
        
    gt = torch.from_numpy(gt)
    pt = torch.from_numpy(pt)

    full_sequences_gt = np.concatenate((input_sequences_np, ground_truths_np), axis=1)
    full_sequences_pred = np.concatenate((input_sequences_np, means_np), axis=1)

    full_ground_truth = inverse_normalise(full_sequences_gt, scaler, scaler_type)
    full_predictions = inverse_normalise(full_sequences_pred, scaler, scaler_type)

    colors = ['teal', 'darkblue', 'indigo', 'orangered', 'darkseagreen']
    colors2 = ['turquoise', 'cornflowerblue', 'mediumpurple', 'lightcoral', 'lightgreen']

    num_series = input_sequences.shape[2]

    data_dict = {
        'input_sequences': input_sequences_np.tolist(),
        'ground_truths': ground_truths_np.tolist(),
        'means': means_np.tolist(),
        'timestamps': timestamps[0, :, 0].tolist(),
        'farm_names': farm_names[0],
    }
    sns.set_theme()
    plt.figure(figsize=(15, 8))
    
    timestamps_flattened = pd.to_datetime(timestamps[0, :, 0].flatten())
    date_str = timestamps_flattened[0].strftime("%Y-%m-%d") 

    for i in range(num_series):
        color = colors[i % len(colors)]
        color2 = colors2[i % len(colors2)]

        series_gt = full_ground_truth[0, :, i]
        series_pred = full_predictions[0, :, i]

        ax = plt.subplot(num_series, 1, i + 1)
        sns.lineplot(x=timestamps_flattened, y=series_pred, label='Prediction', marker='x', color=color2, ax=ax, linewidth='2')
        sns.lineplot(x=timestamps_flattened, y=series_gt, label='Ground Truth', marker='o', color=color, ax=ax, linewidth='2')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='grey')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle='--', linewidth='0.5', color='grey')
        ax.set_xticks(timestamps_flattened[::2])
        ax.set_xlabel("")

        farm_info = f"{farm_names[0][i]} \n{date_str}"
        ax.text(1.05, 0.8, farm_info, transform=ax.transAxes, fontsize=12, verticalalignment='center', 
                bbox=dict(facecolor='white', alpha=0.5))
        
        metrics = calculate_metrics(gt[:, :, i], pt[:, :, i])

        metrics_text = f"""MAE = {metrics['mae']:.5f}
        MSE = {metrics['mse']:.5f}
        RMSE = {metrics['rmse']:.5f}
        R2 = {metrics['r2']:.5f}
        MAPE = {metrics['mape']:.5f}%
        MRE = {metrics['mre']:.5f}
        MBE = {metrics['mbe']:.5f}
        MAAPE = {metrics['maape']:.5f}
        SDE = {metrics['sde']:.5f}"""   
             
        ax.text(1.05, 0.3, metrics_text, transform=ax.transAxes, fontsize=12, verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.5))
        
        sns.despine()  
        ax.legend(loc='center left')
        plt.tight_layout()
        
    save_to_file(data_dict, os.path.join(save_dir, f"{plot_index}.txt"))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(save_dir, f"{plot_index}.png"))  
    plt.close()

def plot_24h_predictions(input_sequences, ground_truths, means, timestamps, farm_names, save_dir, plot_index, scaler, scaler_type):
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1.5})

    input_sequences_np = input_sequences.numpy()  
    ground_truths_np = ground_truths.numpy()  
    means_np = means.numpy()  

    gt = inverse_normalise(ground_truths_np, scaler, scaler_type)
    pt = inverse_normalise(means_np, scaler, scaler_type)
        
    gt = torch.from_numpy(gt)
    pt = torch.from_numpy(pt)

    full_sequences_gt = np.concatenate((input_sequences_np, ground_truths_np), axis=1)
    full_sequences_pred = np.concatenate((input_sequences_np, means_np), axis=1)

    ground_truths = inverse_normalise(full_sequences_gt, scaler, scaler_type)
    predictions = inverse_normalise(full_sequences_pred, scaler, scaler_type)

    colors = ['teal', 'darkblue', 'indigo', 'orangered', 'darkseagreen']
    colors2 = ['turquoise', 'cornflowerblue', 'mediumpurple', 'lightcoral', 'lightgreen']

    plt.figure(figsize=(15, 8))

    consistent_segments = []
    current_segment = []
    for i in range(len(farm_names) - 96):
        current_farms = set(farm_names[i])
        if all(current_farms == set(farm_names[j]) for j in range(i, i + 96)):
            if not current_segment or i == current_segment[-1] + 1:
                current_segment.append(i)
            else:
                if len(current_segment) >= 96:
                    consistent_segments.append(current_segment)
                current_segment = [i]

    for segment in consistent_segments:
        timestamps_segment = timestamps[segment[0]:segment[0]+96]
        input_segment = input_sequences[segment[0]:segment[0]+96]
        ground_truth_segment = ground_truths[segment[0]:segment[0]+96]
        prediction_segment = predictions[segment[0]:segment[0]+96]

        timestamps_flattened = pd.to_datetime(timestamps_segment.flatten())
        sns.lineplot(x=timestamps_flattened, y=prediction_segment.flatten(), label='Prediction', color='blue')
        sns.lineplot(x=timestamps_flattened, y=ground_truth_segment.flatten(), label='Ground Truth', color='red')

    plt.legend()
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"24h_predictions_{plot_index}.png"))
    plt.close()

    return consistent_segments

def plot_results_probabilistic(input_sequences, ground_truths, means, variance, timestamps, farm_names, save_dir, plot_index, scaler, scaler_type):
    import seaborn as sns
    input_sequences_np = input_sequences.numpy()  
    ground_truths_np = ground_truths.numpy()  
    means_np = means.numpy()  
    variance_np = variance.numpy()  

    gt = inverse_normalise(ground_truths_np, scaler, scaler_type)
    pt = inverse_normalise(means_np, scaler, scaler_type)
    var = inverse_normalise(variance_np, scaler, scaler_type) ** 0.5 

    gt = torch.from_numpy(gt)
    pt = torch.from_numpy(pt)

    full_sequences_gt = np.concatenate((input_sequences_np, ground_truths_np), axis=1)
    full_sequences_pred = np.concatenate((input_sequences_np, means_np), axis=1)

    full_ground_truth = inverse_normalise(full_sequences_gt, scaler, scaler_type)
    full_predictions = inverse_normalise(full_sequences_pred, scaler, scaler_type)

    colors = ['teal', 'darkblue', 'indigo', 'orangered', 'darkseagreen']
    colors2 = ['turquoise', 'cornflowerblue', 'mediumpurple', 'lightcoral', 'lightgreen']

    num_series = input_sequences.shape[2]

    data_dict = {
        'input_sequences': input_sequences_np.tolist(),
        'ground_truths': ground_truths_np.tolist(),
        'means': means_np.tolist(),
        'timestamps': timestamps[0, :, 0].tolist(),
        'farm_names': farm_names[0],
    }
    sns.set_theme()
    plt.figure(figsize=(15, 8))
    
    timestamps_flattened = pd.to_datetime(timestamps[0, :, 0].flatten())
    date_str = timestamps_flattened[0].strftime("%Y-%m-%d") 

    for i in range(num_series):
        color = colors[i % len(colors)]
        color2 = colors2[i % len(colors2)]

        series_gt = full_ground_truth[0, :, i]
        series_pred = full_predictions[0, :, i]
        series_var = var[0, :, i]

        ax = plt.subplot(num_series, 1, i + 1)
        sns.lineplot(x=timestamps_flattened, y=series_pred, label='Prediction', marker='x', color=color2, ax=ax, linewidth='2')
        sns.lineplot(x=timestamps_flattened, y=series_gt, label='Ground Truth', marker='o', color=color, ax=ax, linewidth='2')
        
        # Handle variance for last 3 steps
        variance_times = timestamps_flattened[-len(series_var):]
        prediction_end = series_pred[-len(series_var):]
        ax.fill_between(variance_times, prediction_end - 1.96 * series_var, prediction_end + 1.96 * series_var, color=color2, alpha=0.2)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='grey')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle='--', linewidth='0.5', color='grey')
        ax.set_xticks(timestamps_flattened[::2])
        ax.set_xlabel("")

        farm_info = f"{farm_names[0][i]} \n{date_str}"
        ax.text(1.05, 0.8, farm_info, transform=ax.transAxes, fontsize=12, verticalalignment='center', 
                bbox=dict(facecolor='white', alpha=0.5))
        
        metrics = calculate_metrics(gt[:, :, i], pt[:, :, i])

        metrics_text = f"""MAE = {metrics['mae']:.5f}
        MSE = {metrics['mse']:.5f}
        RMSE = {metrics['rmse']:.5f}
        MBE = {metrics['mbe']:.5f}
        MAAPE = {metrics['maape']:.5f}
        """   
             
        ax.text(1.05, 0.3, metrics_text, transform=ax.transAxes, fontsize=12, verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.5))
        
        sns.despine()  
        ax.legend(loc='center left')
        plt.tight_layout()
        
    save_to_file(data_dict, os.path.join(save_dir, f"{plot_index}.txt"))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(save_dir, f"{plot_index}.png"))  
    plt.close()

def plot_input_data(input_seq, ground_truth, sequence_length, prediction_length):
    colors = ['teal', 'darkblue', 'indigo', 'orangered', 'darkseagreen']  
    plt.figure(figsize=(15, 8))
    num_series = input_seq.shape[1]
    total_length = sequence_length + prediction_length
    time_steps = np.arange(total_length)

    for i in range(num_series):
        color = colors[i % len(colors)]
        plt.subplot(num_series, 1, i + 1)
        series_input = input_seq[:, i]
        series_ground_truth = ground_truth[:, i]
        full_series = np.concatenate([series_input, series_ground_truth], axis=0)

        plt.plot(time_steps, full_series, label='Full Series', marker='o', color=color)
        plt.axvline(x=sequence_length - 1, color=color, linestyle='--', label='Prediction Start')
        plt.xlabel('Time Step')
        plt.ylabel(f'Series {i+1} Value')
        plt.title(f'Time Series {i+1}')
        plt.ylim(0, 1)
        plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()