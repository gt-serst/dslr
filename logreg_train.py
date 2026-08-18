import sys
import pandas as pd
import numpy as np
import pickle
from model import gradient_descent
from utils import predict_belongs_to_house, preprocessing

if __name__ == '__main__':
	try:
		if len(sys.argv) == 2:
			df = pd.read_csv(sys.argv[1])
			X, y = preprocessing(df)
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
	except FileNotFoundError as e:
		print("Wrong file or file path:", e)
	except BaseException as e:
		print("An unexpected error occurred:", e)


