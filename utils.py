import numpy as np

def predict_belongs_to_house(X, thetas_array):
	thetas_sum = np.sum(np.multiply(X, thetas_array), axis=1)
	y_pred = 1 / (1 + np.exp(-thetas_sum))
	return y_pred

def preprocessing(df):
	df.columns = df.columns.str.replace(" ", "_").str.lower()
	# Drop not meaningful variables
	df = df.drop(["index", "first_name", "last_name", "birthday"], axis=1)
	# Isolate dependant variable
	y = df["hogwarts_house"]
	X = df.drop("hogwarts_house", axis=1)
	num_features = X.select_dtypes(include=[int, float]).columns
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
	X["bias"] = 1.0
	bias = X["bias"]
	X.drop(labels=["bias"], axis=1, inplace = True)
	X.insert(0, "bias", bias)
	X = X.to_numpy(dtype=float)
	return X, y
