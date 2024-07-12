import torch
import torch.nn as nn
from .attention import MultiSeriesAttention


class MultiSeriesDecoderNoSpatial(nn.Module):
    def __init__(self, hidden_size, num_series, num_layers, num_directions, prediction_length, spatial_feature_size):
        super(MultiSeriesDecoderNoSpatial, self).__init__()
        self.hidden_size = hidden_size
        self.num_series = num_series
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.prediction_length = prediction_length
        self.attention = MultiSeriesAttention(hidden_size, num_directions)
        self.lstm = nn.LSTM(self.num_series + self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size * self.num_directions, self.num_series * self.prediction_length)

    def forward(self, input, hidden, cell, encoder_outputs, spatial_data):
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        hidden = hidden.permute(1, 0, 2, 3).reshape(self.num_layers, batch_size, self.hidden_size * self.num_directions)
        cell = cell.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        cell = cell.permute(1, 0, 2, 3).reshape(self.num_layers, batch_size, self.hidden_size * self.num_directions)
        decoder_hidden_for_attention = hidden[-1]
        attention_weights = self.attention(encoder_outputs, decoder_hidden_for_attention)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
        input_combined = torch.cat((input, context_vector), -1)
        output, (hidden, cell) = self.lstm(input_combined, (hidden, cell))
        point_prediction = self.out(output.squeeze(1)).view(-1, self.num_series, self.prediction_length)
        point_prediction = point_prediction.view(-1, self.num_series, point_prediction.size(-1)).permute(0, 2, 1)
        return point_prediction, None, hidden, cell
