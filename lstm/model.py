import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rnns are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. Rnn
# allow us to operate over sequence of vectors

class RNNModel(nn.Module) :
    def __init__(self , input_dim , hidden_dim , layer_dim , output_dim , dropout_prob) :
        super(RNNModel , self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim , hidden_dim , layer_dim , batch_first=True , dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim , output_dim)

    def forward(self , x) :
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim , x.size(0) , self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out , h0 = self.rnn(x , h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[: , -1 , :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out


class LSTMModel(nn.Module) :
    def __init__(self , input_dim , hidden_dim , layer_dim , output_dim , dropout_prob) :
        super(LSTMModel , self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim , self.hidden_dim , self.layer_dim , batch_first=True , dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim , output_dim)
        #self.relu = nn.ReLU()

        #Softmax
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self , x, future=0) :
        # Initializing hidden state for first input with zeros
        h = torch.zeros(self.layer_dim , x.size(0) , self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c = torch.zeros(self.layer_dim , x.size(0) , self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out , (hn , cn) = self.lstm1(x , (h.detach().to(device) , c.detach().to(device)))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[: , -1 , :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        #out = self.relu(out)
        out = self.fc(out)

        #out = self.softmax(out)

        return out


class GRUModel(nn.Module) :
    def __init__(self , input_dim , hidden_dim , layer_dim , output_dim , dropout_prob) :
        super(GRUModel , self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim , hidden_dim , layer_dim , batch_first=True , dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim , output_dim)

    def forward(self , x) :
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim , x.size(0) , self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out , _ = self.gru(x , h0.detach().to(device))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[: , -1 , :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
