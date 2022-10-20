import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optims
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from customdata import ImageDataset
import argparse
from tensorboardX import SummaryWriter


from net2 import CNNEncoder , RNNDecoder


CUDA_LAUNCH_BLOCKING = "1"
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

df = pd.read_csv(
    '/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Features/cropped/features_crop6.csv')
mapping = {item: i for i , item in enumerate(df["Hr_label"].unique())}
df["Labels"] = df["Hr_label"].apply(lambda x : mapping[x])
train_set , test_set = train_test_split(df , test_size=0.2 , shuffle=False)

img_folder = '/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/croppedFrames/rgb/'
name = 'CNN-LSTM_user6'
model_name  = 'cnnlstm'
output_path = '/home/gabriella/PycharmProjects/HR-predictions/results/'

# Hyper-parameters
num_epochs = 20
batch_size = 32
learning_rate = 0.00001
num_classes = df["Hr_label"].unique().size

def weights_init(m):
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
    transforms.Resize((224, 224)) ,
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

train_dataset_size = len(train_dataloader)
val_dataset_size = len(test_dataloader)
# tensorboard

# model = CNNLSTM()
# model = Combine()
# model = models.vgg16(pretrained=True)
# model = FeatureExtractor(model)
model = nn.Sequential(CNNEncoder() , RNNDecoder(num_classes=num_classes))
model = model.to(device)
print_network(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-06)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

iteration = 0
best_acc = 0.0
for epoch in range(num_epochs):
    with open("result.txt", 'a') as output:
        output.write('Epoch {}/{}'.format(epoch+1, num_epochs) + '\n')
        output.write('-'*10 + '\n')
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-'*10)
    model.train()
    train_loss = 0.0
    train_corrects = 0.0
    val_loss = 0.0
    val_corrects = 0.0
for (image , labels) in train_dataloader :
    iter_loss = 0.0
    iter_corrects = 0.0
    image = image.cuda()
    labels = labels.cuda()
    outputs = model(image)
    _ , preds = torch.max(outputs.data , 1)
    loss = criterion(outputs , labels)
    loss.backward()
    optimizer.step()
    iter_loss = loss.data.item()
    train_loss += iter_loss
    for i in range(len(preds)) :
        if preds[i] == labels.data[i] :
            iter_corrects += 1
    train_corrects += iter_corrects
    iteration += 1
    if (iteration % 5 == 0) :
        with open("result.txt" , 'a') as output :
            output.write('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration , iter_loss / batch_size ,
                                                                              iter_corrects / batch_size) + '\n')
        print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration , iter_loss / batch_size ,
                                                                   iter_corrects / batch_size))
epoch_loss = train_loss / train_dataset_size
epoch_acc = train_corrects / train_dataset_size
with open("result.txt" , 'a') as output :
    output.write('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss , epoch_acc))
print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss , epoch_acc))

model.eval()
with torch.no_grad():
    for (image , labels) in test_dataloader :
        image = image.cuda()
        labels = labels.cuda()
        outputs = model(image)
        _ , preds = torch.max(outputs.data , 1)
        loss = criterion(outputs , labels)
        val_loss += loss.data.item()
        for i in range(len(preds)) :
            if preds[i] == labels.data[i] :
                val_corrects += 1
    epoch_loss = val_loss / val_dataset_size
    epoch_acc = val_corrects / val_dataset_size
    with open("result.txt" , 'a') as output :
        output.write('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss , epoch_acc) + '\n')
    print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    if epoch_acc > best_acc :
        best_acc = epoch_acc
        best_model_wts = model.state_dict()
scheduler.step()
# if not (epoch % 40):
torch.save(model.state_dict(), os.path.join(output_path , str(epoch) + '_' + model_name))
with open("result.txt" , 'a') as output:
    output.write('Best val Acc: {:.4f}'.format(best_acc) + '\n')
print('Best val Acc: {:.4f}'.format(best_acc))
# model.load_state_dict(best_model_wts)
# torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
