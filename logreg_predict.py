import pickle
import pandas as pd
import sys
import numpy as np
from utils import predict_belongs_to_house, preprocessing

if __name__ == '__main__':
	try:
		if len(sys.argv) == 2:
			df = pd.read_csv(sys.argv[1])
			X, y = preprocessing(df)
			with open("thetas_array.pickle", "rb") as f:
				thetas_array = pickle.load(f)
			probabilities = []
			for thetas in thetas_array:
				probability = predict_belongs_to_house(X, thetas)
				probabilities.append(probability)
			predictions = np.argmax(probabilities, axis=0)
			houses = np.array([
				"Ravenclaw",
				"Slytherin",
				"Gryffindor",
				"Hufflepuff"
			])
			pred_df = pd.DataFrame({
				"Index": np.arange(len(predictions)),
				"Hogwarts House": houses[predictions]
			})
			pred_df.to_csv("houses.csv", index=False)
			result = np.column_stack((X, predictions))
			result_df = pd.DataFrame(result)
			result_df.to_csv("result.csv")
	except FileNotFoundError as e:
		print("Wrong file or file path:", e)
	except BaseException as e:
		print("An unexpected error occurred:", e)
