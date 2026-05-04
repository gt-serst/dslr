def compute_error(X, y, houses_to_predict):

	print(y)
	print(houses_to_predict)
	y.loc(y["hogwarts_house"] == houses_to_predict, "hogwarts_house") = 1
	print(1/0)
	m = len(X)
	X["error"] = -1/m * y
