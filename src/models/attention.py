import torch
import torch.nn as nn

class InputAttention(nn.Module):
    def __init__(self, encoder_hidden_size, num_series):
        super(InputAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.num_series = num_series
        self.attention = nn.Linear(self.encoder_hidden_size + self.num_series, self.num_series)

    def forward(self, hidden, encoder_inputs):
        combined = torch.cat((hidden.repeat(encoder_inputs.size(1), 1, 1).permute(1, 0, 2), encoder_inputs), dim=2)
        energy = torch.tanh(self.attention(combined))
        attention_weights = torch.softmax(energy, dim=2)
        return attention_weights

class MultiSeriesAttention(nn.Module):
    def __init__(self, hidden_size, num_directions):
        super(MultiSeriesAttention, self).__init__()
        self.attention_input_size = hidden_size * num_directions + hidden_size * num_directions
        self.attention = nn.Linear(self.attention_input_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(batch_size, seq_len, -1)
        concatenated_inputs = torch.cat((decoder_hidden_expanded, encoder_outputs), dim=2)
        energy = torch.tanh(self.attention(concatenated_inputs))
        attention_weights = torch.softmax(energy, dim=1)
        return attention_weights
