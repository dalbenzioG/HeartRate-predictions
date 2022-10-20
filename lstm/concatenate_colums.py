import pandas as pd
import os
import glob

# Importing Training Set
df1 = pd.read_csv(
    '/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Features/cropped/features_cropped11.csv')

df2 = pd.read_csv(
    '/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Features/raw_thermalframes/features_user11.csv')

labels = df2["Hr_label"]

df1 = df1.join(labels)
df1.to_csv('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Features/cropped/features_crop11.csv', index=False)

