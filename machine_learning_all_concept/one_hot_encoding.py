import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\CAR DETAILS FROM CAR DEKHO.csv",usecols=[0,2,3,4,7]).head(100)
print(df)
print(df.shape)
print(df.isnull().sum())
print(df.describe())

values=df["name"].value_counts()
print(values)
uni_values=df["fuel"].unique()
print(uni_values)
# pd.set_option("display.max_columns", None)
after_encode=pd.get_dummies(df[["fuel","owner"]],drop_first=True)
print(after_encode)

x=df.drop("selling_price",axis=1)
y=df["selling_price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("x train data ",x_train)
ohe=OneHotEncoder(drop="first",dtype=np.int32)
output=ohe.fit(x_train[["fuel","owner"]])
output1=ohe.transform(x_train[["fuel","owner"]]).toarray()
print(output1.shape)
print(output1)