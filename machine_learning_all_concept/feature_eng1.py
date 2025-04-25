import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df =pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Social_Network_Ads2.csv")
print(df)
print(df.columns)
df.columns = df.columns.str.strip()
# df= df.iloc[:,2:]

# df=df.sample(5)
print(df)
X = df[['Age', 'EstimatedSalary']]
Y = df['Purchased']
print(X)
X_train ,X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

print(X_train.shape)

print("hhfhrygfrgrigre")