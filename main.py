import os
import logging
import argparse

from alive_progress import alive_bar
from src.utils.file_operations import generate_parameters_name, extract_config_from_pathname
from src.preprocessing.dataset import load_or_create_dataset
from src.models.models_utils import initialize_models
from scripts.test import test
from scripts.train import train
from scripts.persistence import persistence
from scripts.config_manager import load_common_config, load_train_config, load_test_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Train or test the AttentionRNN model.')
    parser.add_argument('--mode', choices=['train', 'test', 'test_all', 'persistence', 'predict_all'], required=True, help='Select mode: train or test')
    parser.add_argument('--config', required=True, help='Configuration file to use')
    return parser.parse_args()

def main():
    args = parse_args() 
    data_folder, data_folder, sequence_length, sequence_length, prediction_length, num_series, hidden_size, num_layers, num_directions, normalization, cluster_type, config = load_common_config(args.config)

    if args.mode == "train":
        epochs, learning_rate, data_path, loss_function = load_train_config(args.config)
        if config.get('probabilistic', {}).get('probabilistic_use', True):
            loss_function = 'gaussian_negative_log_likelihood'
        data_path = os.path.join(data_folder, data_path) 
        dataset, timestamp, farm_names, scaler, scaler_type, features, spatial_data = load_or_create_dataset(
            data_path,
            sequence_length,
            prediction_length,
            num_series,
            normalization,
            args.mode,
            cluster_type,
            config)         
        
        logging.info("Initializing encoder and decoder models...")
        encoder, decoder = initialize_models(num_series, features, hidden_size, num_layers, num_directions, prediction_length, spatial_data.shape[1], config)
        logging.info("Encoder and decoder models initialized.\n")
        parameters_filename = generate_parameters_name(config)
        parameters_path = os.path.join('parameters/', parameters_filename)
        
        train(dataset, spatial_data, encoder, decoder, num_layers, num_directions, epochs, hidden_size, learning_rate, parameters_filename, loss_function, num_series, config)
        #if not os.path.exists(parameters_path):
        #    train(dataset, spatial_data, encoder, decoder, num_layers, num_directions, epochs, hidden_size, learning_rate, parameters_filename, loss_function, num_series, config)
        #else:
        #    logging.info(f"The parameters file '{parameters_filename}' already exists. Skipping training.")

    if args.mode == "test":
        plot, plot_type, data_path, parameters_path = load_test_config(args.config)
        data_folder, sequence_length, prediction_length, num_series, hidden_size, num_layers, num_directions, normalization, epochs, learning_rate, cluster_type, spatial = extract_config_from_pathname(parameters_path)
        data_path = os.path.join(data_folder, data_path)  
        dataset, timestamp, farm_names, scaler, scaler_type, features, spatial_data = load_or_create_dataset(
            data_path,
            sequence_length,
            prediction_length,
            num_series,
            normalization,
            args.mode,
            cluster_type,
            config) 
   
        logging.info("Initializing encoder and decoder models...")
        encoder, decoder = initialize_models(num_series, features, hidden_size, num_layers, num_directions, prediction_length, spatial_data.shape[1], config)
        logging.info("Encoder and decoder models initialized.\n")
        logging.info("Testing progress started...")
        test(dataset, spatial_data, sequence_length ,timestamp, farm_names, encoder, decoder, scaler, scaler_type, parameters_path, plot, plot_type, num_series, args, config)

    if args.mode == "test_all":
        parameters_folder = 'parameters/'
        for parameter_names in os.listdir(parameters_folder):
            epochs_files_path = os.path.join(parameters_folder, parameter_names)
            for epoch_files in os.listdir(epochs_files_path):
                if epoch_files.endswith('.pt') and 'best_model' in epoch_files:
                    parameters_path = os.path.join(epochs_files_path, epoch_files)
                    parameters_path = str(parameters_path).replace('\\', '/')

                    logging.info(f"Testing {parameters_path}")

                    plot, plot_type, data_path, _ = load_test_config(args.config)
                    data_folder, sequence_length, prediction_length, num_series, hidden_size, num_layers, num_directions, normalization, epochs, learning_rate, cluster_type, spatial = extract_config_from_pathname(parameters_path)
                    data_path = os.path.join(data_folder, data_path)  
                    dataset, timestamp, farm_names, scaler, scaler_type, features, spatial_data = load_or_create_dataset(
                        data_path,
                        sequence_length,
                        prediction_length,
                        num_series,
                        normalization,
                        args.mode,
                        cluster_type,
                        config) 
            
                    logging.info("Initializing encoder and decoder models...")
                    encoder, decoder = initialize_models(num_series, features, hidden_size, num_layers, num_directions, prediction_length, spatial_data.shape[1], config)
                    logging.info("Encoder and decoder models initialized.\n")
                    logging.info("Testing progress started...")
                    
                    parameters_path = os.path.join(parameter_names, epoch_files)
                    test(dataset, spatial_data, sequence_length ,timestamp, farm_names, encoder, decoder, scaler, scaler_type, parameters_path, plot, plot_type, num_series, args, config)

    if args.mode == "persistence":
        plot, plot_type, data_path, parameters_path = load_test_config(args.config)
        data_folder, sequence_length, prediction_length, num_series, hidden_size, num_layers, num_directions, normalization, epochs, learning_rate, cluster_type, spatial = extract_config_from_pathname(parameters_path)
        data_path = os.path.join(data_folder, data_path)  
        dataset, timestamp, farm_names, scaler, scaler_type, features, spatial_data = load_or_create_dataset(
            data_path,
            sequence_length,
            prediction_length,
            num_series,
            normalization,
            'test',
            cluster_type,
            config) 
        
        persistence(dataset)


    if args.mode == "predict_all":
        plot, plot_type, data_path, parameters_path = load_test_config(args.config)
        data_folder, sequence_length, prediction_length, num_series, hidden_size, num_layers, num_directions, normalization, epochs, learning_rate, cluster_type, spatial = extract_config_from_pathname(parameters_path)
        data_path = os.path.join(data_folder, data_path)  
        dataset, timestamp, farm_names, scaler, scaler_type, features, spatial_data = load_or_create_dataset(
            data_path,
            sequence_length,
            prediction_length,
            num_series,
            normalization,
            args.mode,
            cluster_type,
            config) 
   
        logging.info("Initializing encoder and decoder models...")
        encoder, decoder = initialize_models(num_series, features, hidden_size, num_layers, num_directions, prediction_length, spatial_data.shape[1], config)
        logging.info("Encoder and decoder models initialized.\n")
        logging.info("Testing progress started...")
        test(dataset, spatial_data, sequence_length ,timestamp, farm_names, encoder, decoder, scaler, scaler_type, parameters_path, plot, plot_type, num_series, args, config)


if __name__ == "__main__":
    main()