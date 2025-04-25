import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read CSV (ensure correct columns are read)
df = pd.read_csv(r"C:\Users\hp\Downloads\Wine dataset.csv",  usecols=[0, 1, 2]).head(100)

# Remove any spaces from column names
df.columns = df.columns.map(str).str.strip()
print("Columns before renaming:", repr(df.columns))
df.rename(columns={df.columns[0]: "Class", df.columns[1]: "Alcohol", df.columns[2]: "Malicacid"}, inplace=True)

print("Columns after renaming:", repr(df.columns))
print(df.head())

# Scatter plot with correct column names
# sns.scatterplot(x="Alcohol", y="Malicacid", hue="Class", data=df)
# plt.show()
X = df[["Alcohol", "Malicacid"]]  # Features (independent variables)
Y = df["Class"]  # Target (dependent variable)

# X = X.apply(pd.to_numeric, errors='coerce')

X = X.fillna(X.mean())  # Replace NaN values with the mean of the column

print("Independent variables (X):")
print(X)
print("\nDependent variable (Y):")
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

print(X_test_scale)
X_train_scaled_df = pd.DataFrame(X_train_scale, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scale, columns=X.columns)
# print(X_test_scaled_df.describe())
print(np.round(X_train_scaled_df.describe(), decimals=2))
combined_df = pd.concat([X_train_scaled_df, X_test_scaled_df])
x_column = combined_df.columns[0]
y_column = combined_df.columns[1]
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=combined_df, x=x_column, y=y_column, hue='Class',)

# sns.scatterplot(data=result,x=X_train_scaled_df,y=X_test_scaled_df, fill=True, color="blue")
# plt.show()