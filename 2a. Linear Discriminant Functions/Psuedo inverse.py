### Pseudo Inverse for MSE
import math
import numpy as np

# Input data - WITH BIAS
X = np.array([
    [1, 0,2],
    [1, 1,2],
    [1, 2,1],
    [1, -3,1],
    [1, -2,-1],
    [1, -3,-2]
], dtype = np.float)

y = np.array([
    1,1,1,-1,-1,-1
], dtype = np.float)

# Set weights - WITH BIAS
b = np.array(
    [1,1,1,2,2,2], 
    dtype = np.float)


### Perform sample normalisation ###
for i in range(len(X)):
    if y[i] < 0:
        X[i] = X[i] * -1

print(X)

# Calculate the pseduo inverse

W = np.dot(np.linalg.pinv(X), b)

print("W:")
print(np.round(W, 4))