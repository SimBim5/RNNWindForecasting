# |█░░██░░██    ░░██░░                      ████████╗░██╗░░██╗░██╗░░░██╗░██╗░██╗░░░██╗░░██████╗░
# |░██░░██░░██░░██░░██░░██░░██░░██          ╚══██╔══╝░██║░░██║░███╗░░██║░██║░███╗░░██║░██╔════╝░
# |█░░██░░██░░██░░██░░██░░██░░██░░██░       ░░░██║░░░░██║░░██║░██║██╗██║░██║░██║██╗██║░██║░████╗
# |░██░░██░░██░░██░░██░░██░░██░░██░░█       ░░░██║░░░░██║░░██║░██║░████║░██║░██║░████║░██║░░░██║
# |█░░██░░██░░██░░██░░██░░██░░██░░██░       ░░░██║░░░░░█████║░░██║░░███║░██║░██║░░███║░░██████╔╝
# |░██░░██░░    ██░░██░░██░░██░░██░░█       ░░░╚═╝░░░░░╚════╝░░╚═╝░░╚══╝░╚═╝░╚═╝░░╚══╝░░╚═════╝░
# |                 ░░██░░██░░██░░██░
# |                               ░░█
# |                  
# |   Welcome to the Hyperparameter Tuning Garage!
# |   Where models are optimized for peak performance! 
import itertools
import json
import subprocess
import os
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

hyperparameters = {
    "num_series": [6]
}

def has_training_been_done(base_config, train_config, base_path='simon-faltz/parameters'):
    folder_name = f"params_seq{base_config['sequence_length']}_pred3_series5_hidden{base_config['hidden_size']}_layers{base_config['num_layers']}_directions2_normrobust_minmax_normalize_epochs{train_config['epochs']}_lr{train_config['learning_rate']}_clusternearest_mean_squared_error"
    folder_path = os.path.join('/home/simon.faltz/Desktop/simon-faltz/parameters', folder_name)
    return os.path.exists(folder_path)

def run_training(config):
    with open('config_temp.json', 'w') as f:
        json.dump(config, f)

    process = subprocess.Popen(['python', 'main.py', '--mode', 'train', '--config', 'config_temp.json'], stdout=sys.stdout, stderr=sys.stderr, text=True)
    process.wait()

total_combinations = len(list(itertools.product(*hyperparameters.values())))
current_combination = 1

for index, values in enumerate(itertools.product(*hyperparameters.values())):
    config_variation = dict(zip(hyperparameters.keys(), values))
    
    with open('config.json', 'r') as base_config_file:
        base_config = json.load(base_config_file)
    
    for key, value in config_variation.items():
        if key in base_config['common']:
            base_config['common'][key] = value
        elif key in base_config['train']:
            base_config['train'][key] = value

    if has_training_been_done(base_config['common'], base_config['train']):
        logging.info(f"Skipping training {current_combination} / {total_combinations} as it's already done.")
    else:
        logging.info(f"Running training {current_combination} / {total_combinations}.")
        logging.info(f"Configuration: {config_variation}")
        run_training(base_config)
    current_combination += 1

if os.path.exists('config_temp.json'):
    os.remove('config_temp.json')
