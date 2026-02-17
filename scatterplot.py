import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
	df = pd.read_csv("dataset_train.csv")
	df.columns = df.columns.str.replace(" ", "_").str.lower()
	df = df.dropna()
	num_cols = df.select_dtypes(include=float).columns
	for col_1 in num_cols:
		for col_2 in num_cols:
			if col_1 == col_2:
				continue
			sns.scatterplot(data=df, x=col_1, y=col_2, hue='hogwarts_house')
			plt.title(f'Scatterplot of {col_1} and {col_2}')
			plt.show()
