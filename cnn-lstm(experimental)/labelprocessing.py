import pandas as pd

df = pd.read_csv('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent '
                 'System/cropped/target/features_6.csv')

mapping = {item:i for i, item in enumerate(df["Hr_label"].unique())}
df["Labels"] = df["Hr_label"].apply(lambda x: mapping[x])

