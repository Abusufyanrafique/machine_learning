import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder , LabelEncoder
from feature_eng1 import X_test, Y_train

df= pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Encoding_data.csv")
print(df)
print(df.columns)
df.columns = df.columns.str.strip()

print("null values",df.isnull().sum())
# if df.shape[1] >= 3:
#     df = df.iloc[:, 0:3]  # Select first 3 columns
# else:
#     df = df.iloc[:, 0:]
# df.loc[:, ['Review', 'Education', 'Purchased']] = df[['Review', 'Education', 'Purchased']].fillna(df.mode().iloc[0])
# print(df)

X_train,x_test,y_train,Y_test = train_test_split(df.iloc[:,2:4],df.iloc[:,-1],test_size=0.2)
print("My x train ",X_train)
print("My y train ",y_train)
# oe=OrdinalEncoder(categories=[["Good quality "," Not satisfied","Adequate","Poor quality","Very happy",
#                                "Could be better",
#                                " Loved it",
#                                " Not as expected",
#                                "  Highly recommend"
#                                ],["Master","Bachelor","PhD"]]
#
#                   )
oe=OrdinalEncoder()
oe.fit(X_train)
encode_values=oe.transform(X_train)
print("encode_values",encode_values)

le=LabelEncoder()
le.fit(Y_train)
y_train=le.transform(Y_train)
print(y_train)

le.fit(Y_test)
y_test=le.transform(Y_test)
print(y_test)





