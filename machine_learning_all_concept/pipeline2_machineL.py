import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\tested_titanic (1).csv")
print("Original DataFrame columns:", df.columns)
# if df==df.colums:
#     print("have complete items")
# else:
#     print("Noting is here///////////////")

df = df.drop(columns=["Cabin", "PassengerId", "Ticket", "Name"])
print("DataFrame after dropping columns:", df.columns)

print("Missing values in each column:\n", df.isnull().sum())

x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=["Survived"]), df["Survived"], test_size=0.2)

num_columns = ['Age', 'SibSp', 'Parch', 'Fare']
cat_columns = ['Pclass', 'Sex', 'Embarked']

print("Numeric columns:", num_columns)
print("Categorical columns:", cat_columns)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("Imputer", SimpleImputer(strategy="mean")),
            ("Scaling", MinMaxScaler())
        ]), num_columns),
        ("cat", Pipeline([
            ("Imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_columns)
    ]
)

pipe = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier())
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
