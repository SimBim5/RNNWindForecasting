import torch
import torch.nn as nn
import warnings
from .attention import InputAttention

class MultiSeriesEncoder(nn.Module):
    def __init__(self, num_series, hidden_size, num_layers, num_directions, prediction_length, dropout_rate=0.5):
        if num_layers == 1 and dropout_rate > 0:
            warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=%.1f and num_layers=%d" % (dropout_rate, num_layers), category=UserWarning)
        super(MultiSeriesEncoder, self).__init__()
        self.num_series = num_series
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.prediction_length = prediction_length
        self.lstm = nn.LSTM(num_series, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.input_attention = InputAttention(hidden_size, num_series)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        input_attention_weights = self.input_attention(h0[-1], x)
        attended_input = input_attention_weights * x
        outputs, (hidden, cell) = self.lstm(attended_input, (h0, c0))
        return outputs, hidden
