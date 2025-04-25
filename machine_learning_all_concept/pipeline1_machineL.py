import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline ,make_pipeline
from sklearn.feature_selection import chi2
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\tested_titanic (1).csv")
print(df)
print(df.isnull().sum())
print(df.columns)
print(df.shape)

df = df.drop(columns=["Cabin","PassengerId","Ticket","Name"])
print("After Droping the columns ", df)
# print("",df.columns)
x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=["Survived"])
                                               ,df["Survived"],test_size=0.2)
print("x train data",x_train.shape)
print("y train data ",y_train)
print(df.columns)
trf1=ColumnTransformer(
    transformers=[
        ("impute_age",SimpleImputer(),[2]),
        ("impute_embarked",SimpleImputer(strategy="most_frequent"),[6])
    ],remainder="passthrough"
)
trf2=ColumnTransformer(
    transformers=[
        ("ohe_sex_embarked",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),[1,6])
    ],remainder="passthrough"
)

trf3=ColumnTransformer(
    transformers=[
        ("scale",MinMaxScaler(),slice(0,7))
    ]
)

trf4=SelectKBest(score_func=chi2,k=5)

trf5=DecisionTreeClassifier()

pipe=Pipeline([
    ("trf1",trf1),
    ("trf2",trf2),
    ("trf3",trf3),
    ("trf4",trf4),
    ("trf5",trf5)
])

pipe=make_pipeline(trf1,trf2,trf3,trf4,trf5)
result=pipe.fit(x_train,y_train)
print(result)
diagram = set_config(display="diagram")
print(diagram)

y_predict=pipe.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)