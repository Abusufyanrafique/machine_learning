import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

print("Hello world")

df= pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Social_Network_Ads2.csv")
print(df)
print(df.shape)
X = df[["Age", "EstimatedSalary"]]
Y=df["Purchased"]

print("columns of independent ",X)
print("columns of dependent ",Y)

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train)
print(Y_train)

scaler= StandardScaler()
result=scaler.fit(X_train)
print(scaler.mean_)
print(X_train.describe())
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)
print("result2",X_test_scaled)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=["Age", "EstimatedSalary"])
print(X_train_scaled_df.describe())
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=["Age", "EstimatedSalary"])
print(X_test_scaled_df.describe())

sns.scatterplot(x=X_train["Age"], y=X_train["EstimatedSalary"])
plt.figure(figsize=(8,6))
# before scaling
sns.scatterplot(x=X_train_scaled_df["Age"], y=X_train_scaled_df["EstimatedSalary"])
plt.show()

# Train model on this data ////////////////////////////////////////////////////////////////////////////////////////

lr=LogisticRegression()
before_scaling=lr.fit(X_train,Y_train)
predicted_values=lr.predict(X_test)
print("Before scaling",predicted_values)

lr=LogisticRegression()
after_scaling = lr.fit(X_train_scaled_df,Y_train)
predicted_values1=lr.predict(X_test_scaled_df)
print("After scaling",predicted_values1)

print("Before scaling", accuracy_score(Y_test , predicted_values))
print("After  scaling", accuracy_score(Y_test , predicted_values1))
# Decision tree model use
dt=DecisionTreeClassifier()
before_scaling1=dt.fit(X_train,Y_train)
predicted_values2=dt.predict(X_test)

after_scaling2=dt.fit(X_train_scaled_df,Y_train)
predicted_values3=dt.predict(X_test_scaled_df)

print("Before scaling", accuracy_score(Y_test , predicted_values))
print("After  scaling", accuracy_score(Y_test , predicted_values1))

# Affect of outliers in the data
df = pd.concat([df, pd.DataFrame({
"Age":[5,90,95],
"EstimatedSalary":[1000,250000,350000],
"purchased":[0,1,1],
})], ignore_index=True)
sns.scatterplot(x="Age", y="EstimatedSalary", data=df)

plt.show()