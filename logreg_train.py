import sys
import pandas as pd
import numpy as np
import pickle
from model import gradient_descent
from utils import predict_belongs_to_house

if __name__ == '__main__':

	# try:
		if len(sys.argv) == 2:
			df = pd.read_csv(sys.argv[1])
			df.columns = df.columns.str.replace(" ", "_").str.lower()
			# Drop not meaningful variables
			df = df.drop(["index", "first_name", "last_name", "birthday"], axis=1)
			# Isolate dependant variable
			y = df["hogwarts_house"]
			print(y.value_counts())
			X = df.drop("hogwarts_house", axis=1)
			num_features = X.select_dtypes(include=[int, float]).columns
			cat_features = X.select_dtypes(include=str).columns
			# Handle NA values
			cols = X.isna().any()
			cols_with_na = cols[cols].index
			for col in cols_with_na:
				X[col] = X[col].fillna(X[col].mean())
			# Perform Label Encoding
			X.loc[X["best_hand"] == "Left", "best_hand"] = str(1)
			X.loc[X["best_hand"] == "Right", "best_hand"] = str(0)
			X["best_hand"] = X["best_hand"].astype(int)
			# Standardize numerical features
			for col in num_features:
				X[col] = (X[col] - X[col].mean()) / X[col].std()
			# Add a bias
			bias_series = pd.Series(1)
			X["bias"] = 1.0
			bias = X["bias"]
			X.drop(labels=["bias"], axis=1, inplace = True)
			X.insert(0, "bias", bias)
			X = X.to_numpy(dtype=float)
			# thetas_array = []
			# for i in enumerate(X.columns):
			# 	thetas_array.append(0)
			# Initialize thetas to 0
			thetas_array = np.zeros((4, X.shape[1]))
			alpha = 1
			epochs = 10000
			batch_size = len(X)
			# We will loop over the houses to predict but start with the first one
			houses_to_predict = y.unique()
			for i, house in enumerate(houses_to_predict):
				print(f"House: {house}")
				y_binary = np.where(y == house, 1, 0)
				# Launch gradient descent optimization
				thetas_array[i] = gradient_descent(X, y_binary, alpha, epochs, batch_size, thetas_array[i])
				y_pred = predict_belongs_to_house(X, thetas_array[i])
				df["y_pred"] = y_pred
				df.to_excel(f"pred_{house.lower()}.xlsx")
			with open('thetas_array.pickle', 'wb') as f:
				pickle.dump(thetas_array, f, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			raise BaseException("program must take one argument")
	# except FileNotFoundError as e:
	# 	print("Wrong file or file path:", e)
	# except BaseException as e:
	# 	print("An unexpected error occurred:", e)


