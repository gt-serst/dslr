import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

	try:
		df = pd.read_csv("dataset_train.csv")
		df.columns = df.columns.str.replace(" ", "_").str.lower()
		df = df[df==df]
		df.drop(['index'], inplace=True, axis=1)
		num_cols = df.select_dtypes(include=[int, float]).columns
		df = df[["hogwarts_house"] + list(num_cols)]
		sns.pairplot(df, hue="hogwarts_house", corner=True)
		plt.show()
	except FileNotFoundError as e:
		print("Wrong file or file path:", e)
	except BaseException as e:
		print("An unexpected error occurred:", e)
