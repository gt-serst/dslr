import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
	df = pd.read_csv("dataset_train.csv")
	df.columns = df.columns.str.replace(" ", "_").str.lower()
	df = df.dropna()
	num_cols = df.select_dtypes(include=float).columns
	df = df[["hogwarts_house"] + list(num_cols)]
	sns.pairplot(df, hue="hogwarts_house", corner=True)
	plt.show()
