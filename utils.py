import numpy

def predict_belongs_to_house(X, thetas_array):
	thetas_sum = X.mul(thetas_array, axis="columns").sum(axis="columns")
	X["pred"] = 1 / (1 + numpy.exp(-thetas_sum))
	print(X["pred"])
