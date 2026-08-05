import numpy as np

def predict_belongs_to_house(X, thetas_array):
	thetas_sum = np.sum(np.multiply(X, thetas_array), axis=1)
	y_pred = 1 / (1 + np.exp(-thetas_sum))
	return y_pred
