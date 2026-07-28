from utils import predict_belongs_to_house
import numpy as np

def compute_error(X, y, y_pred):
	eps = 1e-15
	y_pred = np.clip(y_pred, eps, 1 - eps)
	m = len(X)
	cost = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
	return cost

def calculate_thetas_gradient(alpha, thetas_array, X, y_binary, y_pred, max_iterations=10000):
	m = len(X)
	for _ in range(max_iterations):
		gradient_thetas = []
		# Loop over thetas
		for i in range(len(thetas_array)):
			# Calculate each gradient
			gradient = (1/m) * np.sum((y_pred - y_binary) * X.iloc[:, i])
			gradient_thetas.append(gradient)
		epsilon = 1e-6
		# Check if all gradients are close to zero means a minimun is found
		if all(abs(i) < epsilon for i in gradient_thetas):
			break
		else:
			# If not need to update thetas (move a little to find if the next position could be the minimun)
			thetas_array = update_thetas(alpha, thetas_array, gradient_thetas)
			# Predict again
			y_pred = predict_belongs_to_house(X, thetas_array)
	return thetas_array, y_pred

def update_thetas(alpha, thetas_array, gradient_thetas):
	return thetas_array - alpha * gradient_thetas
