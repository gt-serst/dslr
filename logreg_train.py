import sys
import pandas as pd
import numpy as np

if __name__ == '__main__':

	try:
		if len(sys.argv) == 2:
			df = pd.read_csv(sys.argv[1])
			df.columns = df.columns.str.replace(" ", "_").str.lower()
			# Drop not meaningful variables
			df = df.drop(["index", "first_name", "last_name", "birthday"], axis=1)
			# Isolate dependant variable
			y = df["hogwarts_house"]
			X = df.drop("hogwarts_house", axis=1)
			num_features = X.select_dtypes(include=[int, float]).columns
			cat_features = X.select_dtypes(include=str).columns
			# Handle NA values
			cols = df.isna().any()
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
			print(X)
		else:
			raise BaseException("program must take one argument")
	except FileNotFoundError as e:
		print("Wrong file or file path:", e)
	except BaseException as e:
		print("An unexpected error occurred:", e)


