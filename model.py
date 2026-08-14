from utils import predict_belongs_to_house
import numpy as np

def compute_error(X, y_binary, y_pred):
	eps = 1e-15
	y_pred = np.clip(y_pred, eps, 1 - eps)
	m = len(X)
	cost = -1/m * np.sum(y_binary * np.log(y_pred) + (1 - y_binary) * np.log(1 - y_pred))
	return cost

def calculate_gradient(y_pred, y_batch, X_batch):
	m = len(X_batch)
	return (1/m) * X_batch.T.dot(y_pred - y_batch)

def gradient_descent(X, y_binary, alpha, epochs, batch_size, thetas_array):
	m = len(X)
	epsilon = 1e-6
	previous_cost = None
	cost_history = []
	for epoch in range(epochs):
		# Shuffle the dataset
		indices = np.random.permutation(m)
		X_shuffled = X[indices]
		y_shuffled = y_binary[indices]
		# Mini-batches
		for i in range(0, m, batch_size):
			X_batch = X_shuffled[i:i + batch_size]
			y_batch = y_shuffled[i:i + batch_size]
			# Prediction for this batch
			y_pred_batch = predict_belongs_to_house(X_batch, thetas_array)
			# Gradient for this batch
			gradient_thetas = calculate_gradient(y_pred_batch, y_batch, X_batch)
			# Update theta
			thetas_array = update_thetas(alpha, thetas_array, gradient_thetas)
		# Evaluate the model on the complete dataset
		y_pred = predict_belongs_to_house(X, thetas_array)
		cost = compute_error(X, y_binary, y_pred)
		cost_history.append(cost)
		if epoch % 100 == 0:
			print(f"Epoch {epoch}, Cost: {cost}")
		if previous_cost is not None:
			if abs(previous_cost - cost) < epsilon:
				break
		previous_cost = cost
	return thetas_array

def update_thetas(alpha, thetas_array, gradient_thetas):
	return thetas_array - alpha * gradient_thetas
