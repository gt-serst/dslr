from utils import predict_belongs_to_house
import numpy as np

def compute_error(X, y_binary, y_pred):
	eps = 1e-15
	y_pred = np.clip(y_pred, eps, 1 - eps)
	m = len(X)
	cost = -1/m * np.sum(y_binary * np.log(y_pred) + (1 - y_binary) * np.log(1 - y_pred))
	return cost

def calculate_gradient(y_pred, y_binary, X):
	m = len(X)
	return (1/m) * X.T.dot(y_pred - y_binary)

def gradient_descent(alpha, thetas_array, X, y_binary, max_iterations=10000):
	epsilon = 1e-6
	previous_cost = None
	cost_history = []
	for iteration in range(max_iterations):
		# Predict with sigmoid function
		y_pred = predict_belongs_to_house(X, thetas_array)
		# Compute error between prediction and target vector with logloss function
		cost = compute_error(X, y_binary, y_pred)
		if iteration % 100 == 0:
			print(iteration, cost)
		cost_history.append(cost)
		if previous_cost is not None:
			# Stop if the cost is no longer decreasing significantly
			if abs(previous_cost - cost) < epsilon:
				break
		previous_cost = cost
		gradient_thetas = calculate_gradient(y_pred, y_binary, X)
		# If not need to update thetas (move a little to find if the next position could be the minimun)
		thetas_array = update_thetas(alpha, thetas_array, gradient_thetas)
	return thetas_array

def update_thetas(alpha, thetas_array, gradient_thetas):
	return thetas_array - alpha * gradient_thetas
