import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    
def load_common_config(config):
    config = load_config(config)
    common_config = config['common']
    data_folder = common_config['data_folder']
    sequence_length = common_config['sequence_length']
    prediction_length = common_config['prediction_length']
    num_series = common_config['num_series']
    hidden_size = common_config['hidden_size']
    num_layers = common_config['num_layers']
    num_directions = common_config['num_directions']
    normalization = common_config['normalization']
    cluster_type = common_config['cluster_type']
    return data_folder, data_folder, sequence_length, sequence_length, prediction_length, num_series, hidden_size, num_layers, num_directions, normalization, cluster_type, config

def load_test_config(config):
    config = load_config(config)
    test_config = config['test']
    plot = test_config['plot']
    plot_type = test_config['plot_type']
    data_path = test_config['data_path']
    parameters_path = test_config['parameters_path']
    return plot, plot_type, data_path, parameters_path

def load_train_config(config):
    config = load_config(config)
    train_config = config['train']
    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    data_path = train_config['data_path']
    loss_function = train_config['loss_function']
    return epochs, learning_rate, data_path, loss_function

def load_spatial_config(config):
    config = load_config(config)
    spatial_config = config['spatial']
    plot = spatial_config['plot']
    return plot