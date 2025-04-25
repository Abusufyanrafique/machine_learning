import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder,FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline ,make_pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import scipy.stats as stats

from pipeline2_machineL import y_pred

df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\tested_titanic (1).csv",usecols=["Survived","Age","Fare"])
print(df)
print(df.isnull().sum())
print(df.columns)

df["Age"]= df["Age"].fillna(df["Age"].mean(),inplace=False)
print(df)
print("typeh",type(df))
print("Missing values",df.isnull().sum())

x= df[['Age','Fare']]
y=df['Survived']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
print("values of x train",x_train)
x_train = x_train.fillna(x_train.mean())
x_test = x_test.fillna(x_test.mean())

# plt.figure(figsize=(12,8))
# # sns.distplot(x_train["Age"])
# sns.distplot(x_train["Fare"])
# stats.probplot(x_train["Fare"], dist="norm", plot=plt)
plt.title("Probability Plot for Age")
plt.show()
#
# clf1=LogisticRegression()
# clf2=DecisionTreeClassifier()
# clf1.fit(x_train,y_train)
# clf2.fit(x_train,y_train)
# predict1=clf1.predict(x_test)
# predict2=clf2.predict(x_test)
# print(predict1)
# print(predict2)
#
# accuracy=accuracy_score(y_test,predict1)
# print(accuracy)
# accuracy1=accuracy_score(y_test,predict2)
# print("accuracy1", accuracy1)

trf=FunctionTransformer(func=np.log1p)
x_train_transformed=trf.fit_transform(x_train)
x_test_transformed=trf.transform(x_test)
clf1=LogisticRegression()
clf2=DecisionTreeClassifier()

clf1.fit(x_train_transformed,y_train)
clf2.fit(x_train_transformed,y_train)
y_pred=clf1.predict(x_test_transformed)
y_pred=clf2.predict(x_test_transformed)

accuracy3=accuracy_score(y_test,y_pred)
accuracy4=accuracy_score(y_test,y_pred)
print(accuracy3)
print(accuracy4)

