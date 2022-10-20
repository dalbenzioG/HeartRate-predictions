import pandas as pd
import os
import glob


# path = os.getcwd()
# csv_files = glob.glob(os.path.join('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/Features/cropped/', "*.csv"))
#
# li = []
# # loop over the list of csv files
# for f in sorted(csv_files):
#     # read the csv file
#     df = pd.read_csv(f)
#     li.append(df)
#     print(csv_files)
#
# result = pd.concat(li, axis=0, ignore_index=True)
# result.to_csv('path to csv file', index=False)


df1 = pd.read_csv(
    'path to csv file')

df2 = pd.read_csv(
    'path to csv file')

result = pd.concat([df1,df2], axis=0, ignore_index=True)
result.to_csv('path to csv file', index=False)

