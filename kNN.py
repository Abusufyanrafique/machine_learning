import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\breast-cancer.csv")

print(df)
df.drop(columns=["id"],inplace=True)
print(df.shape)
Y = df["diagnosis"]
Y = Y.map({'M': 1, 'B': 0}) 
df.columns = df.columns.str.strip()
X = df[["radius_mean", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]]
print("Y.head",Y.head(100))
print("X head",X.head(100))
X_train,X_test, Y_train,Y_test= train_test_split(X,Y, test_size=0.2)

print(X_train.shape)

st = StandardScaler()
scaler = st.fit_transform(X_train)
st.transform(X_test)

print(scaler)
for i in range(1,16):
        knn = KNeighborsClassifier(n_neighbors=i)
knn.fit(X_train,Y_train)

predicted_val = knn.predict(X_test)
print("X test",X_test)
score = accuracy_score(Y_test,predicted_val)

print("Accuracy:", score)
print("predicted values",predicted_val)

plt.plot(range(1, 16), score, marker='o')
plt.show()