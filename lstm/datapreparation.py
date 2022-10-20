import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , RobustScaler

batch_size=128

def get_scaler(scaler) :
    scalers = {
        "minmax" : MinMaxScaler ,
        "standard" : StandardScaler ,
        "maxabs" : MaxAbsScaler ,
        "robust" : RobustScaler ,
    }
    return scalers.get(scaler.lower())()


scaler = get_scaler('robust')

# Importing Training Set data = pd.read_csv( '/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for
# Intelligent System/Features/cropped/features_6.csv')

#data = pd.read_csv('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Features/cropped/allfeatures.csv')
data = pd.read_csv('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Features/cropped/features_crop6.csv')
#data = pd.read_csv('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Features/cropped/features1_2.csv')

# Select features (columns) to be involved intro training and predictions
training_set = data.iloc[: , 1:].values

# plt.plot(training_set, label = 'Shampoo Sales Data')
plt.plot(training_set[:,88] , label=' Features first_order User6')
plt.show()

# features
df_features = data.iloc[:, 1:]

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, y_train, y_val

X_train, X_val, y_train, y_val = train_val_test_split(df_features, 'Hr_label', 0.2)
plt.plot(y_val.iloc[:,:].values , label=' Features first_order User6')
plt.show()

scaler = get_scaler("minmax")
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
#X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
#y_test_arr = scaler.transform(y_test)

#
train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr)
val_targets = torch.Tensor(y_val_arr)
#test_features = torch.Tensor(X_test_arr)
#test_targets = torch.Tensor(y_test_arr)

train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
#test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
#test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader_one = DataLoader(val, batch_size=1, shuffle=False, drop_last=True)

