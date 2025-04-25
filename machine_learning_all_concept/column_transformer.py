import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from unicodedata import category

df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\covid1.csv")
print(df)

print(df.isnull().sum())
x=df.drop(columns=["hospitalization"])
y=df["hospitalization"]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=42)
print("my x train data ",x_train)
print("y train data",y_train)
# train data
imputer1 =SimpleImputer(strategy='most_frequent')
result=imputer1 .fit_transform(x_train)
print("use imputer class",result)
print("use imputer class",result.shape)
# test data
result1=imputer1 .fit_transform(x_test)
print("use imputer class",result1)
print("use imputer class",result1.shape)

le=LabelEncoder()
encode_value=le.fit_transform(df["hospitalization"])
print(encode_value.shape)
      # ordinal data

# imputer = SimpleImputer(strategy='most_frequent')  # or 'constant' with 'unknown' if needed
# x_train["fever"] = imputer.fit_transform(x_train[["fever"]])
#
# # Now apply OrdinalEncoder after handling missing values
# oe = OrdinalEncoder()
# values = oe.fit_transform(x_train["fever"])
# print("use ordinal encoding ",values)
# use transformer /////////////////////////////////////////////////////////
preprocessor=ColumnTransformer(
    transformers=[
         ("tnf1",SimpleImputer(strategy='most_frequent'),["fever","cough","fatigue"]),
        ("tnf2",StandardScaler(),["age"])

    ],remainder="passthrough"
)
transformed_data=preprocessor.fit_transform(x_train)
print(transformed_data)

