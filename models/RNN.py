import torch
import torch.nn as nn

def get_rnn_model(model_type, input_size, hidden_size, num_layers, dropout):
    if model_type == 'LSTM':
        model = LSTMModel(input_size, hidden_size, num_layers, dropout)
    elif model_type == 'GRU':
        model = GRU(input_size, hidden_size, num_layers, dropout)
    else:
        raise NotImplementedError('Wrong model type')
    return model


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # same as self.linear.forward(out[:, -1, :])

        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(GRU, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])  # same as self.linear.forward(out[:, -1, :])

        return out




