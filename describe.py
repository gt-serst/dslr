import pandas as pd
import numpy as np
import math
import sys

def count(series):
	return len(series)

def mean(series):
	return sum(series) / len(series)

def std(series):
	return np.sqrt(sum(pow(series - sum(series) / len(series), 2)) / (len(series) - 1))

def min(series):
	min_value = series.iloc[0]
	for value in series:
		if value < min_value:
			min_value = value
	return min_value

def percentile_25(series):
	series = series.sort_values().reset_index(drop=True)
	rank = (25 / 100) * (len(series) - 1)
	floor_rank = math.floor(rank)
	ceil_rank = math.ceil(rank)
	if floor_rank == ceil_rank:
		percentile = series.iloc[floor_rank]
	else:
		percentile = series.iloc[floor_rank] + (series.iloc[ceil_rank] - series.iloc[floor_rank]) * (rank - floor_rank)
	return percentile

def percentile_50(series):
	series = series.sort_values().reset_index(drop=True)
	rank = (50 / 100) * (len(series) - 1)
	floor_rank = math.floor(rank)
	ceil_rank = math.ceil(rank)
	if floor_rank == ceil_rank:
		percentile = series.iloc[floor_rank]
	else:
		percentile = series.iloc[floor_rank] + (series.iloc[ceil_rank] - series.iloc[floor_rank]) * (rank - floor_rank)
	return percentile

def percentile_75(series):
	series = series.sort_values().reset_index(drop=True)
	rank = (75 / 100) * (len(series) - 1)
	floor_rank = math.floor(rank)
	ceil_rank = math.ceil(rank)
	if floor_rank == ceil_rank:
		percentile = series.iloc[floor_rank]
	else:
		percentile = series.iloc[floor_rank] + (series.iloc[ceil_rank] - series.iloc[floor_rank]) * (rank - floor_rank)
	return percentile

def max(series):
	max_value = series.iloc[0]
	for value in series:
		if value > max_value:
			max_value = value
	return max_value

def count_na(series):
	return len(series[series!=series])

def describe(series, col):
	series_keeping_na_value = series
	series = series[series==series]
	df = pd.DataFrame(index=[col])
	df.loc[col, "count"], df.loc[col, "mean"], df.loc[col, "std"], df.loc[col, "min"], df.loc[col, "25%"], df.loc[col, "50%"], df.loc[col, "75%"], df.loc[col, "max"], df.loc[col, "na"] = count(series), mean(series), std(series), min(series), percentile_25(series), percentile_50(series), percentile_75(series), max(series), count_na(series_keeping_na_value)
	return df

if __name__ == '__main__':

	try:
		df = pd.read_csv("dataset_train.csv")
		df.columns = df.columns.str.replace(" ", "_").str.lower()
		num_cols = df.select_dtypes(include=[int, float]).columns
		describe_df = pd.DataFrame()
		for col in list(num_cols):
			describe_df = pd.concat([describe_df, describe(df[col], col)])
		describe_df = describe_df.T
		if describe_df.empty is False:
			print(describe_df)
		else:
			print("Dataframe do not have numerical features.")
		print(df.describe())
	except FileNotFoundError as e:
		print("Wrong file or file path:", e)
	except Exception as e:
		print("An unexpected error occurred:", e)
