import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Apply the default theme
sns.set_theme()


data = pd.read_csv('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Results/1User/result_batch32.csv')

#data['time(s)'] = data.index

sns.lineplot(data=data)
plt.show()
