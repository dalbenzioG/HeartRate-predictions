import torch
import torch.nn as nn
import torch.optim as optim

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision import models

from net2 import CNNEncoder , RNNDecoder
from train import train_epoch
from torch.utils.data import DataLoader
from customdata import ImageDataset
from utils import Optimization , calculate_accuracy
from torch.utils.tensorboard import SummaryWriter
from vgglstm import FeatureExtractor
from validation import val_epoch
from model import CNNLSTM , Combine

CUDA_LAUNCH_BLOCKING = "1"
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_STORE_PATH = '/home/gabriella/gabri/results/'

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

df = pd.read_csv(
    'path to csv file')
# mapping = {item : i for i , item in enumerate(df["Hr_label"].unique())}
# df["Labels"] = df["Hr_label"].apply(lambda x : mapping[x])
train_set , test_set = train_test_split(df , test_size=0.2 , shuffle=False)

img_folder = 'path to rgb'

# Hyper-parameters
n_features = 1
log_interval = 20
num_epochs = 10
batch_size = 4
learning_rate = 0.0001


def weights_init(m) :
    if isinstance(m , nn.Conv2d) :
        torch.nn.init.kaiming_uniform_(m.weight.data , a=0 , mode='fan_in')
    elif isinstance(m , nn.BatchNorm2d) :
        torch.nn.init.normal_(m.weight.data , 1.0 , 0.02)
        torch.nn.init.constant_(m.bias.data , 0.0)


def print_network(net) :
    num_params = 0
    for param in net.parameters() :
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def resume_model(opt , model , optimizer) :
    checkpoint = torch.load(opt.resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model Restored from Epoch {}".format(checkpoint['epoch']))
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def format_predictions(predictions , values , df_test) :
    vals = np.concatenate(values , axis=0).ravel()
    preds = np.concatenate(predictions , axis=0).ravel()
    df_result = pd.DataFrame(data={"value" : vals , "prediction" : preds} , index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    return df_result

def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'mse': mean_squared_error(df.value, df.prediction),
            'r2' : r2_score(df.value, df.prediction)}


train_transform = transforms.Compose([
    transforms.ToPILImage() ,
    transforms.Resize((224 , 224)) ,
    transforms.RandomHorizontalFlip(p=0.5) ,
    transforms.RandomRotation(degrees=45) ,
    transforms.ToTensor(),
    transforms.Normalize([0.5436867 , 0.18738046 , 0.4981859] , [0.2641301 , 0.22807583 , 0.22663283])
    ])

test_transform = transforms.Compose([
    transforms.ToPILImage() ,
    transforms.Resize((224 , 224)) ,
    transforms.ToTensor(),
    transforms.Normalize([0.5436867 , 0.18738046 , 0.4981859] , [0.2641301 , 0.22807583 , 0.22663283])
    ])

train_dataset = ImageDataset(train_set , img_folder , train_transform)
test_dataset = ImageDataset(test_set , img_folder , test_transform)

train_dataloader = DataLoader(
    train_dataset ,
    batch_size=batch_size ,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset ,
    batch_size=1,
    shuffle=False
)

total_step = len(train_dataloader)
# tensorboard

# model = CNNLSTM()
# model = Combine()
# model = models.vgg16(pretrained=True)
# model = FeatureExtractor(model)
model = nn.Sequential(CNNEncoder() , RNNDecoder())
model = model.to(device)
print_network(model)
# criterion = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters() , lr=learning_rate , betas=(0.9 , 0.999) , eps=1e-08 , weight_decay=1e-5)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max=len(train_dataloader) , eta_min=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer , mode='min' , factor=0.1 , threshold=0.01 , patience=10)
model.apply(weights_init)

opt = Optimization(model=model , loss_fn=loss_fn , optimizer=optimizer)
opt.train(train_dataloader , test_dataloader , batch_size=batch_size , n_epochs=num_epochs)
opt.plot_losses()

predictions , values = opt.evaluate(test_dataloader , batch_size=1)

df_result = format_predictions(predictions , values , test_dataloader)

result_metrics = calculate_metrics(df_result)

import matplotlib.pyplot as plt
import seaborn as sns
ax = plt.gca()
df_result.plot(kind='line',y='value', use_index=True, ax=ax)
df_result.plot(kind='line',y='prediction', use_index=True, ax=ax)
plt.show()

# for epoch in range(1 , num_epochs + 1) :
#     print('Epoch {}/{}'.format(epoch + 1 , num_epochs))
#     print('-' * 10)
#     # start training
#     for i , (data , targets) in enumerate(train_dataloader) :
#
#         data , targets = data.to(device) , targets.to(device)
#         #print(data.shape)
#         #print(targets.shape)
#         data = data.to(torch.float32)
#         targets = targets.to(torch.float32)
#
#         # print(data.shape)
#
#         outputs = model(data)
#
#         loss = criterion(targets, outputs)
#         losses.append(loss)
#
#
#
#
