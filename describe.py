import pandas as pd

df = pd.read_csv("dataset_train.csv")
df.columns = df.columns.str.replace(" ", "_").str.lower()
print(df.dtypes)
print(df)
print(df.describe())
