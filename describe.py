import pandas as pd
import numpy as np
import math

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

def describe(series, col):
	series = series.dropna()
	df = pd.DataFrame(index=[col])
	df.loc[col, "count"], df.loc[col, "mean"], df.loc[col, "std"], df.loc[col, "min"], df.loc[col, "25%"], df.loc[col, "50%"], df.loc[col, "75%"], df.loc[col, "max"] = count(series), mean(series), std(series), min(series), percentile_25(series), percentile_50(series), percentile_75(series), max(series)
	return df

df = pd.read_csv("dataset_train.csv")
df.columns = df.columns.str.replace(" ", "_").str.lower()
print(df.dtypes)
print(df)
print(df.describe())
num_cols = df.select_dtypes(include=float).columns
describe_df = pd.DataFrame()
for col in ["index"] + list(num_cols):
	describe_df = pd.concat([describe_df, describe(df[col], col)])
describe_df = describe_df.T
print(describe_df)
