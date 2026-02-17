import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
	df = pd.read_csv("dataset_train.csv")
	df.columns = df.columns.str.replace(" ", "_").str.lower()
	df = df.dropna()
	num_cols = df.select_dtypes(include=float).columns
	for col in num_cols:
		sns.histplot(data=df, x=col, hue='hogwarts_house')
		plt.title(f'Histogram of {col} distribution')
		plt.show()

