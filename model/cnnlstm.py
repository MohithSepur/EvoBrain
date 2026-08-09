import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import utils 

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=1, dataset='TUSZ', in_channels=19, in_dim=100):
        super(CNN_LSTM, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Dynamically calculate flattened feature dimension for fc1
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, in_dim)
            dummy_out = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy)))))
            flat_dim = dummy_out.numel()

        self.fc1 = nn.Linear(flat_dim, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, seq_lengths):

        batch, max_seq_len, num_ch, in_dim = x.shape
        x_flat = x.reshape(-1, num_ch, in_dim).unsqueeze(1)

        out = F.relu(self.conv1(x_flat))
        out = F.relu(self.conv2(out))
        out = self.pool(out)

        out = out.reshape(batch * max_seq_len, -1)

        assert out.shape[1] == self.fc1.in_features, (
            f"Input feature size {out.shape[1]} does not match expected {self.fc1.in_features}. "
            f"Please instantiate CNN_LSTM with in_channels={num_ch} and in_dim={in_dim}."
        )

        out = F.relu(self.fc1(out))
        out = out.reshape(batch, max_seq_len, -1)

        lstm_out, _ = self.lstm(out)
        lstm_out = utils.last_relevant_pytorch(lstm_out, seq_lengths, batch_first=True)
        logits = self.fc2(lstm_out)

        return logits, lstm_out