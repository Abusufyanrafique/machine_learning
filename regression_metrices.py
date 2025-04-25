import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,  mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\regression1_metrics.csv")
print(df)

df.drop("Unnamed: 2", axis=1, inplace=True)
df.columns = df.columns.str.strip()  
X=df[["cgpa"]]
# X = df[["cgpa"]] 
print(X)
y = df[["package"]]
print("target variable",y)
print(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train data ",X_train,)
print("y_test data ",y_test) 

lr = LinearRegression()
model = lr.fit(X_train,y_train)
# model= lr.predict(X_test)
model= lr.predict([[7.42]])

print(model)

print("Slope (m):", lr.coef_) 

plt.scatter(X_train, y_train, color="blue", label="Training Data") 
plt.scatter(X_test, y_test, color="red", label="Actual Test Data")
plt.plot(X_test, model, color="green", linewidth=2, label="Regression Line")
plt.xlabel("CGPA")
plt.ylabel("Package")
plt.title("Linear Regression: CGPA vs Package")
plt.legend()

plt.show()
print("train data ", X_train)
print ("test data ", y_test)
print("MAE", mean_absolute_error(y_test,y_pred=model))
print("MSE", mean_squared_error(y_test,y_pred=model))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred=model)))
print("r2_score", r2_score(y_test,y_pred=model))
# Coefficient of Model
print("lr.coef_ data",lr.coef_)
# intercept of model 
print("lr.intercept__ data",lr.intercept_)

b0 = model.intercept_ 
print(b0)