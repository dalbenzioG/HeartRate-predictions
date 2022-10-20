import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18 , resnet101


class CNNLSTM(nn.Module) :
    def __init__(self , num_classes=21) :
        super(CNNLSTM , self).__init__()
        self.resnet = resnet101(pretrained=True)
        # modules = list(self.resnet.children())[:-1]  # delete the last fc layer.
        # self.resnet = nn.Sequential(*modules)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features , 300))
        self.lstm = nn.LSTM(input_size=300 , hidden_size=256 , num_layers=3 , batch_first=True , dropout=0.5)
        self.fc1 = nn.Linear(256 , 128)
        # self.fc2 = nn.Linear(128 , num_classes)
        self.fc2 = nn.Linear(128 , 1) #if mse

    def forward(self , x) :
        batch_size, C , H , W = x.size()
        print('x before CNN shape{}'.format(x.shape))
        hidden = torch.zeros(3, x.size(0) , 256).requires_grad_()
        x = self.resnet(x)
        print('x after RNN reshape shape{}'.format(x.shape))
        # To pass it to LSTM, input must be of the from (seq_len, batch, input_size)
        x = x.reshape(batch_size, 128, 128)
        print('x after reshape shape{}'.format(x.shape))
        out , hidden = self.lstm(x.unsqueeze(0) , hidden)

        out = self.fc1(out[-1 , : , :])
        #x = F.relu(x)
        out = self.fc2(out)

        return out



# 3D convolution for sequences of images --> consider the sequences of images as volume

class CNN(nn.Module) :
    def __init__(self , num_classes=21) :
        super(CNN , self).__init__()
        self.conv1 = nn.Conv2d(3 , 8 , kernel_size=5)
        self.conv2 = nn.Conv2d(8 , 24 , kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256 , 128)
        self.fc2 = nn.Linear(128 , num_classes)

    def forward(self , x) :
        x = F.relu(F.max_pool2d(self.conv1(x) , 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)) , 2))
        x = x.view(-1 , 256)
        x = F.relu(self.fc1(x))
        x = F.dropout(x , training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x


class Combine(nn.Module) :
    def __init__(self , num_classes=21) :
        super(Combine , self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=300 ,
            hidden_size=256 ,
            num_layers=3 ,
            batch_first=True)
        self.fc1 = nn.Linear(256 , 128)
        self.fc2 = nn.Linear(128 , num_classes)

    def forward(self , x) :
        batch_size, C , H , W = x.size()
        c_in = x.view(batch_size, C , H , W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(1, batch_size, C*H*W )
        r_out , (h_n , h_c) = self.rnn(r_in)
        r_out2 = self.fc1(r_out[: , -1 , :])

        return F.log_softmax(r_out2 , dim=1)


# def forward(img_seq_batch, hidden):
#     # Loop over each image in the sequences
#     for img_batch in img_seq_batch:
#         # Push batch of images through CNN
#         encoded = cnn(img_batch)
#         # Push CNN output through LSTM (just one time step)
#         output, hidden = lstm(encoded, hidden)
#     # Return the last output/hidden of the LSTM
#     return output, hidden

class Identity(nn.Module) :
    def __init__(self) :
        super(Identity , self).__init__()

    def forward(self , x) :
        return x


class SubUnet_orig(nn.Module) :
    def __init__(self , hidden_size , n_layers , dropt , bi , N_classes=21) :
        super(SubUnet_orig , self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = n_layers

        dim_feats = 4096

        self.cnn = models.alexnet(pretrained=True)
        self.cnn.classifier[-1] = Identity()
        self.rnn = nn.LSTM(
            input_size=dim_feats ,
            hidden_size=self.hidden_size ,
            num_layers=self.num_layers ,
            dropout=dropt ,
            bidirectional=True)
        self.n_cl = N_classes
        if (True) :
            self.last_linear = nn.Linear(2 * self.hidden_size , self.n_cl)
        else :
            self.last_linear = nn.Linear(self.hidden_size , self.n_cl)

    def forward(self , x) :

        batch_size , timesteps , C , H , W = x.size()
        c_in = x.view(batch_size * timesteps , C , H , W)

        c_out = self.cnn(c_in)

        r_out , (h_n , h_c) = self.rnn(c_out.view(-1 , batch_size , 4096))

        r_out2 = self.last_linear(r_out)

        return r_out2