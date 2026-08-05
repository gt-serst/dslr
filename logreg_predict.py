import pickle
import pandas as pd
import sys
import numpy as np
from utils import predict_belongs_to_house

if __name__ == '__main__':
	# try:
		if len(sys.argv) == 2:
			df = pd.read_csv(sys.argv[1])
			df.columns = df.columns.str.replace(" ", "_").str.lower()
			df = df.drop(["index", "first_name", "last_name", "birthday"], axis=1)
			y = df["hogwarts_house"]
			X = df.drop("hogwarts_house", axis=1)
			num_features = X.select_dtypes(include=[int, float]).columns
			cat_features = X.select_dtypes(include=str).columns
			cols = X.isna().any()
			cols_with_na = cols[cols].index
			for col in cols_with_na:
				X[col] = X[col].fillna(X[col].mean())
			X.loc[X["best_hand"] == "Left", "best_hand"] = str(1)
			X.loc[X["best_hand"] == "Right", "best_hand"] = str(0)
			X["best_hand"] = X["best_hand"].astype(int)
			for col in num_features:
				X[col] = (X[col] - X[col].mean()) / X[col].std()
			bias_series = pd.Series(1)
			X["bias"] = 1.0
			bias = X["bias"]
			X.drop(labels=["bias"], axis=1, inplace = True)
			X.insert(0, "bias", bias)
			X = X.to_numpy(dtype=float)
			print(X)
			with open("thetas_array.pickle", "rb") as f:
				thetas_array = pickle.load(f)
			probabilities = []
			for thetas in thetas_array:
				probability = predict_belongs_to_house(X, thetas)
				probabilities.append(probability)
			predictions = np.argmax(probabilities, axis=0)
			houses = np.array([
				"Ravenclaw",
				"Slytherin",
				"Gryffindor",
				"Hufflepuff"
			])
			pred_df = pd.DataFrame({
				"Index": np.arange(len(predictions)),
				"Hogwarts House": houses[predictions]
			})
			pred_df.to_csv("houses.csv", index=False)
			# result = np.column_stack((X, index))
			# result_df = pd.DataFrame(result)
			# result_df.to_csv("result.csv")
	# except FileNotFoundError as e:
	# 	print("Wrong file or file path:", e)
	# except BaseException as e:
	# 	print("An unexpected error occurred:", e)
