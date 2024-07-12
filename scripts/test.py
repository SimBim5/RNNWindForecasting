import os
import logging
import torch

from alive_progress import alive_bar
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import calculate_metrics, update_metrics, log_metrics
from src.evaluation.plotter import plot_results, plot_results_probabilistic
from src.utils.file_operations import generate_test_filename_from_config, prepare_data_for_excel, update_excel
from scripts.config_manager import load_config
from src.utils.transformations import inverse_normalise

def test(dataset, spatial_data, sequence_length ,timestamp, farm_names, encoder, decoder, scaler, scaler_type, parameters_path, plot, plot_type, num_series, args, config):
    parameters_folder = 'parameters/'
    parameters_path = os.path.join(parameters_folder, parameters_path)

    logging.info("Testing process has started.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_params = torch.load(parameters_path, map_location=device)
    encoder.load_state_dict(model_params['encoder_state_dict'])
    decoder.load_state_dict(model_params['decoder_state_dict'])

    metrics = {
        'normalized': { 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': [], 'MAPE': [], 'MRE': [], 'MBE': [], 'MAAPE': [], 'SDE': []},
        'inv_normalized': { 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': [], 'MAPE': [], 'MRE': [], 'MBE': [], 'MAAPE': [], 'SDE': []}
    }

    dataset = TensorDataset(dataset.tensors[0], dataset.tensors[1], spatial_data)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    filename = generate_test_filename_from_config(load_config(args.config))
    save_dir = os.path.join("results/plot", filename)
    if plot: os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        with alive_bar(len(data_loader), bar="smooth", spinner="waves", length=85) as test_bar:
            for i, (input_sequences, ground_truth, spatial_datas) in enumerate(data_loader):
                input_sequences, ground_truth = input_sequences.to(device), ground_truth.to(device)
                
                encoder_outputs, hidden = encoder(input_sequences)
                decoder_input = torch.zeros(input_sequences.size(0), 1, decoder.num_series, device=device)
                cell_state = torch.zeros(encoder.num_layers, 1, encoder.hidden_size * encoder.num_directions)
                point_predictions, variance, _, _ = decoder(decoder_input, hidden, cell_state, encoder_outputs, spatial_datas)
                
                for z in range(num_series):
                    normalized_metrics = calculate_metrics(
                        ground_truth[:, :, z], point_predictions[:, :, z])
                    update_metrics(metrics['normalized'], **normalized_metrics)

                gt = inverse_normalise(ground_truth.numpy()  , scaler[i], scaler_type)
                pt = inverse_normalise(point_predictions.numpy()  , scaler[i], scaler_type)
                gt = torch.from_numpy(gt)
                pt = torch.from_numpy(pt)
                
                
                for z in range(num_series):
                    inv_normalized_metrics = calculate_metrics(gt[:, :, z], pt[:, :, z])
                    update_metrics(metrics['inv_normalized'], **inv_normalized_metrics)

                '''
                if plot:
                    if config.get('probabilistic', {}).get('probabilistic_use', True):
                        batch_timestamps = timestamp[i * data_loader.batch_size : (i + 1) * data_loader.batch_size]
                        batch_farm_names = farm_names[i * data_loader.batch_size : (i + 1) * data_loader.batch_size]

                        input_sequences = input_sequences[:, :sequence_length, :]

                        plot_results_probabilistic(input_sequences[:, :, :num_series], ground_truth, point_predictions, variance, batch_timestamps, 
                            batch_farm_names, save_dir, i+1, scaler[i], scaler_type) 
                    
                    else:
                    batch_timestamps = timestamp[i * data_loader.batch_size : (i + 1) * data_loader.batch_size]
                    batch_farm_names = farm_names[i * data_loader.batch_size : (i + 1) * data_loader.batch_size]

                    input_sequences = input_sequences[:, :sequence_length, :]
                    plot_results(input_sequences[:, :, :num_series], ground_truth, point_predictions, batch_timestamps, 
                        batch_farm_names, save_dir, i+1, scaler[i], scaler_type) 
                '''
                
                test_bar()

    logging.info(f"Testing process completed.")
    overall_means = log_metrics(metrics)
    logging.info(f"Saving results in model_results.xlsx.")
    data_df = prepare_data_for_excel(parameters_path, overall_means)
    update_excel(data_df)
    logging.info(f"Results saved! :-)")