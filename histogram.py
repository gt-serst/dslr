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
		for col in num_cols:
			sns.histplot(data=df, x=col, hue='hogwarts_house')
			plt.title(f'Histogram of {col} distribution')
			plt.show()
	except FileNotFoundError as e:
		print("Wrong file or file path:", e)
	except BaseException as e:
		print("An unexpected error occurred:", e)
