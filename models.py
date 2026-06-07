import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        mean = out.mean(dim=1)      # ou out[:, -1, :] ou out.mean(dim=1)
        #out = self.dropout(out)
        pooled = (last+mean)/2
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class ComplexCNN(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(ComplexCNN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # ??? (batch_size, input_size, sequence_length) ??
        x = self.tcn(x)
        x = torch.mean(x, dim=2)  # ??????
        x = self.fc(x)
        return x
