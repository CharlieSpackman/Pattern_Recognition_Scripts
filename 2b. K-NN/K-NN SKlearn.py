# K-NN SKlearn

# Import modules
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Import Iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create new unseen vectors
s_1 = np.array([[7.8, 2.5, 1.3, 0.2]])
s_2 = np.array([[7.6, 2.9, 5.2, 1.4]])
s_3 = np.array([[6.5, 4.0, 4.8, 1.1]])
s_4 = np.array([[6.0, 2.0, 5.8, 1.3]])
s_5 = np.array([[5.2, 3.8, 4.8, 2.5]])

# Create K-NN with K=1
K_1 = KNeighborsClassifier(n_neighbors=1, metric = "euclidean")
K_1.fit(X, y)

# Predict classes for K=1
print("Predictions for K=1")
print(K_1.predict(s_1)[0])
print(K_1.predict(s_2)[0])
print(K_1.predict(s_3)[0])
print(K_1.predict(s_4)[0])
print(K_1.predict(s_5)[0])




# Create K-NN with K=5
K_5 = KNeighborsClassifier(n_neighbors=5, metric = "euclidean")
K_5.fit(X, y)

# Predict classes for K=1
print("\nPredictions for K=5")
print(K_5.predict(s_1)[0])
print(K_5.predict(s_2)[0])
print(K_5.predict(s_3)[0])
print(K_5.predict(s_4)[0])
print(K_5.predict(s_5)[0])
