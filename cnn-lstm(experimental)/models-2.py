import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim=32, drop_prob=0.3, bn_momentum=0.01):
        '''
        Use the pre-trained model provided by pytorch as the encoder
        '''
        super(CNNEncoder, self).__init__()

        self.cnn_out_dim = cnn_out_dim
        self.drop_prob = drop_prob
        self.bn_momentum = bn_momentum

        # Use the resnet pre-training model to extract features and
        # remove the last layer of classifier
        pretrained_cnn = models.resnet152(pretrained=True)
        cnn_layers = list(pretrained_cnn.children())[:-1]

        # Remove the last fc layer of resnet to extract features
        self.cnn = nn.Sequential(*cnn_layers)
        self.linear = nn.Linear(pretrained_cnn.fc.in_features, 1024)
        self.bn = nn.BatchNorm1d(1024, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.cnn(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class RNNDecoder(nn.Module):
    def __init__(self, use_gru=True, cnn_out_dim=32, rnn_hidden_layers=3, rnn_hidden_nodes=256,
            num_classes=1, drop_prob=0.3):
        super(RNNDecoder, self).__init__()

        self.rnn_input_features = cnn_out_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes

        self.drop_prob = drop_prob
        self.num_classes = num_classes # Adjust the number of categories here

        # rnn configuration parameters
        rnn_params = {
            'input_size': self.rnn_input_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True
        }

        # Use lstm or gru as the rnn layer
        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        # rnn layer output to linear classifier
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_hidden_nodes, 128),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        b, f = x_rnn.shape
        x_rnn = x_rnn.reshape(b, 32, 32)
        rnn_out, _ = self.rnn(x_rnn, None)
        # Note that when the rnn module is defined earlier, batch_first=True guarantees the following structure:
        # rnn_out shape: (batch, timestep, output_size)
        # h_n and h_c shape: (n_layers, batch, hidden_size)
        x = self.fc(rnn_out[:, -1, :])  # Only extract the last layer for output

        return x