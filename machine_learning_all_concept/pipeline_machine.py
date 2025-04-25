import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

df= pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\tested_titanic (1).csv")
print(df)
print(df.isnull().sum())
pd.set_option('display.max_columns', None)
print(df.head(100))
df.drop(columns=["PassengerId","Cabin","Name","Ticket"],inplace=True)
print("after drop column",df)
x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=["Survived"])
                                               ,df["Survived"],test_size=0.2)
print("x train data",x_train)

si_age=SimpleImputer()
x_train_age=si_age.fit_transform(x_train[["Age","Fare"]])
print("x train age",x_train_age.shape)
print("x train age",x_train_age)
# test data /////////////////

x_test_age=si_age.fit_transform(x_test[["Age","Fare"]])
print("x test age",x_test_age.shape)
# print("x test age",x_test_age)

ohe = OneHotEncoder(drop="first",sparse_output=False )
output1=ohe.fit_transform(x_train[["Sex","Embarked"]])
print("output",output1)
# Test data//////////////////////////
ohe = OneHotEncoder(drop="first",sparse_output=False )
output1=ohe.fit_transform(x_test[["Sex","Embarked"]])
print("output_test",output1)

x_train_rem=x_train.drop(columns=["Embarked","Sex","Age"])
x_test_rem=x_test.drop(columns=["Embarked","Sex","Age"])
x_train_transformed=np.concatenate((x_train_rem,))
