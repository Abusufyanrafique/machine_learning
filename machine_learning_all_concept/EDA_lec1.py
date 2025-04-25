import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Titanic-Dataset.csv")
print(df)

# shape_of_data=df.shape
# print(shape_of_data)
# # some first columns
# first_column=df.head()
# print(first_column)

# find random rows

# print(df.sample(5))
# see information about dataset
# print(df.info())
# # find missing values
# print(df.isnull().sum())
# # mathematical representation of data
# print(df.describe())
# # find duplicated values
# print(df.duplicated().sum())
# print(df.drop_duplicates())

# print(df.corr()["Survived"])

# data exprolarty analysis

# sns.countplot(df["Survived"])
# values=df["Pclass"].value_counts().plot(kind="pie", autopct ="%.2f")
# print(values)
print(df.columns)
# sns.histplot(df["Age"],bins=30)
# sns.displot(df["Age"])
print(df["Age"].min())
print(df["Age"].max())
print(df["Age"].skew())
# sns.boxplot(df["Fare"])
plt.show()

