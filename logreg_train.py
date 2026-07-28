import sys
import pandas as pd
import numpy as np
from utils import predict_belongs_to_house
from model import compute_error, calculate_thetas_gradient

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
			# thetas_array = []
			# for i in enumerate(X.columns):
			# 	thetas_array.append(0)
			# Initialize thetas to 0
			thetas_array = np.zeros(len(X.columns))
			alpha = 1
			# We will loop over the houses to predict but start with the first one
			houses_to_predict = y.unique()
			print(houses_to_predict[0])
			y_binary = np.where(y == houses_to_predict[0], 1, 0)
			# Predict with sigmoid function
			y_pred = predict_belongs_to_house(X, thetas_array)
			# Compute error between prediction and target vector with logloss function
			cost = compute_error(X, y_binary, y_pred)
			print(cost)
			# Compute gradient and update thetas
			thetas_array, y_pred = calculate_thetas_gradient(alpha, thetas_array, X, y_binary, y_pred)
			print(houses_to_predict[0])
			print(df["hogwarts_house"])
			df["y_pred"] = y_pred
			print(y_pred)
			df.to_excel("pred.xlsx")
		else:
			raise BaseException("program must take one argument")
	# except FileNotFoundError as e:
	# 	print("Wrong file or file path:", e)
	# except BaseException as e:
	# 	print("An unexpected error occurred:", e)


