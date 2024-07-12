import os
import torch
import logging
import src.training.loss_function as lf

from torch.utils.data import DataLoader, TensorDataset
from alive_progress import alive_bar

    
def train(dataset, spatial_data, encoder, decoder, num_layers, num_directions, epochs, hidden_size, learning_rate, parameters_filename, loss_function, num_series, config):

    train_dataset, val_dataset, train_spatial_data, val_spatial_data = train_val_split(dataset, spatial_data, train_ratio=0.9)
    
    parameters_path = os.path.join('parameters/', parameters_filename)
    if not os.path.exists(parameters_path):
        os.makedirs(parameters_path)
    logging.info("Parameters will be saved under %s", parameters_path)
    
    best_model_path = None
    best_loss = float('inf')
    num_epochs = int(epochs)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), learning_rate)
    criterion = getattr(lf, loss_function)()
        
    dataset = TensorDataset(train_dataset.tensors[0], train_dataset.tensors[1], train_spatial_data)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    val_dataset = TensorDataset(val_dataset.tensors[0], val_dataset.tensors[1], val_spatial_data)
    val_data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    logging.info(f"Starting Training Session. Epochs: {epochs}. Learning rate: {learning_rate}.")

    with alive_bar(len(data_loader) * num_epochs, bar="smooth", spinner="waves", length=85) as train_bar:
        for epoch in range(num_epochs):
            total_loss = 0
            for input_sequences, ground_truth, spatial_datas in data_loader:
                optimizer.zero_grad()
                batch_size = input_sequences.shape[0]
                encoder_outputs, hidden = encoder(input_sequences)
                cell_state = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
                decoder_input = torch.zeros(batch_size, 1, num_series)
                if config.get('probabilistic', {}).get('probabilistic_use', True):
                    point_predictions, variance, hidden, cell_state = decoder(decoder_input, hidden, cell_state, encoder_outputs, spatial_datas)
                    loss = criterion(point_predictions, variance, ground_truth)
                else:
                    point_predictions, _, hidden, cell_state = decoder(decoder_input, hidden, cell_state, encoder_outputs, spatial_datas)
                    loss = criterion(point_predictions.squeeze(1), ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                train_bar()
            
            average_loss = total_loss / len(data_loader)
            val_loss = validate(encoder, decoder, val_data_loader, criterion, num_layers, num_directions, num_series, hidden_size, config)

            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(parameters_path, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_loss': average_loss,
                    'validation_loss': val_loss,
                }, checkpoint_path)

            if val_loss < best_loss:
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_loss = val_loss
                best_model_path = os.path.join(parameters_path, f"best_model_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_loss': average_loss,
                    'validation_loss': val_loss,
                }, best_model_path)
            logging.info(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.8f}, Validation Loss: {val_loss:.8f}')

    logging.info("Training process completed.")
    
    
def train_val_split(dataset, spatial_data, train_ratio):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    main_tensors = [t[train_indices] for t in dataset.tensors]
    val_main_tensors = [t[val_indices] for t in dataset.tensors]
    
    train_dataset = TensorDataset(*main_tensors)
    val_dataset = TensorDataset(*val_main_tensors)

    if isinstance(spatial_data, torch.Tensor):
        train_spatial_data = spatial_data[train_indices]
        val_spatial_data = spatial_data[val_indices]
    else:
        raise TypeError("spatial_data must be a torch.Tensor")

    return train_dataset, val_dataset, train_spatial_data, val_spatial_data


def validate(encoder, decoder, val_loader, criterion, num_layers, num_directions, num_series, hidden_size, config, device='cpu'):
    encoder.eval()
    decoder.eval()
    
    total_val_loss = 0.0
    with torch.no_grad():
        for input_sequences, ground_truth, spatial_datas in val_loader:
            input_sequences, ground_truth, spatial_datas = input_sequences.to(device), ground_truth.to(device), spatial_datas.to(device)
            encoder_outputs, hidden = encoder(input_sequences)
            cell_state = torch.zeros(num_layers * num_directions, input_sequences.size(0), hidden_size).to(device)
            decoder_input = torch.zeros(input_sequences.size(0), 1, num_series).to(device)  # Adjust as per your input requirements
            
            if config.get('probabilistic', {}).get('probabilistic_use', True):
                point_predictions, variance, hidden, cell_state = decoder(decoder_input, hidden, cell_state, encoder_outputs, spatial_datas)
                loss = criterion(point_predictions, variance, ground_truth)
            else:
                point_predictions, _, hidden, cell_state = decoder(decoder_input, hidden, cell_state, encoder_outputs, spatial_datas)
                loss = criterion(point_predictions.squeeze(1), ground_truth)
                
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    
    return avg_val_loss
