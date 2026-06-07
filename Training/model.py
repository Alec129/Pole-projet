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
