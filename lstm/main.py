import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from lstm.datapreparation import X_train, X_val, train_loader, val_loader, val_loader_one, scaler
import os
from lstm.utility import get_model, Optimization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
input_dim = len(X_train.columns)
output_dim = 1
hidden_dim = 128
layer_dim = 3
batch_size = 128
dropout = 0.5
n_epochs = 800
learning_rate = 0.0001
weight_decay = 1e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = get_model('gru', model_params)
model = model.to(device)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

predictions, values = opt.evaluate(val_loader_one, batch_size=1, n_features=input_dim)

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


df_result = format_predictions(predictions, values, X_val, scaler)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'mse': mean_squared_error(df.value, df.prediction),
            'r2' : r2_score(df.value, df.prediction)}

result_metrics = calculate_metrics(df_result)

import matplotlib.pyplot as plt
import seaborn as sns
ax = plt.gca()
df_result.plot(kind='line',y='value', use_index=True, ax=ax)
df_result.plot(kind='line',y='prediction', use_index=True, ax=ax)
plt.show()




