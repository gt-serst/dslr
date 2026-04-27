import sys
import pandas as pd

if __name__ == '__main__':

	try:
		if len(sys.argv) == 2:
			df = pd.read_csv(sys.argv[1])
			print(df)
			cols = df.isna().any()
			cols_with_na = cols[cols].index
			print(cols_with_na)
			for col in cols_with_na:
				df[col] = df[col].fillna(df[col].mean())
			print(df.select_dtypes(include=[str]).columns)
			print(df.dtypes)
			df = df.drop(["First Name", "Last Name", "Birthday"], axis=1)
			df = pd.get_dummies(df, dtype=int)
			print(df)
		else:
			raise BaseException("program must take one argument")
	except FileNotFoundError as e:
		print("Wrong file or file path:", e)
	except BaseException as e:
		print("An unexpected error occurred:", e)


