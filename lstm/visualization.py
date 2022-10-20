import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    '/home/gabriella/PycharmProjects/rabia/heart_rate/results1.csv')

# gru = df.iloc[1000:2800, :]
# gru = gru.reset_index()
# gru['average-frames'] = gru.prediction.rolling(window=24).mean()

#new = gru.groupby(np.arange(len(gru))//24).mean()

# Apply the default theme
sns.set_style("darkgrid")


sns.lineplot(data =df, x = 'index', y = 'loss', linestyle="dashed", alpha  = 1)
#sns.lineplot(data =df, x = 'index', y = 'prediction', linestyle="dashed", alpha  = 0.3)
#sns.lineplot(data =df, x = 'index', y = 'average-frames', alpha  = 1)
plt.xlabel("Epochs")
plt.ylabel("CELoss")
plt.legend(labels=['loss'])
plt.title("Training ResNet152+CNN")
plt.show()