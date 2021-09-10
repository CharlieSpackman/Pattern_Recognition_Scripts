### Sequential Widrow-Hoff
import math
import numpy as np

# Input data - WITH BIAS
X = np.array([
    [1,0,2],
    [1,1,2],
    [1,2,1],
    [1,-3,1],
    [1,-2,-1],
    [1,-3,-2]
], dtype = np.float)

y = np.array([
    1,1,1,-1,-1,-1
], dtype = np.float)

# Set weights - WITH BIAS
W = np.array([1, 0, 0], dtype = np.float)

# Set margin 
b = np.array([
    1,2.5,2.5,2.5,0.5,1
], dtype = np.float)

# Set learning parameters
rate = 0.1
iterations = 2


### Perform sample normalisation ###
for i in range(len(X)):
    if y[i] < 0:
        X[i] = X[i] * -1


print("Before training")
print("W: {}\n".format(W))

# Start iterating
for epoch in range(iterations):

    print("\nIteration: {}".format(epoch+1))

    batch_error = np.array([0,0], dtype = np.float32)

    # Loop through all examples
    for j in range(len(X)):

        # Caculate prediction
        y_hat = np.dot(W, X[j].T)

        # Update weights
        W = W + rate * X[j] * (b[j] - y_hat)

        print(f"Sample: {X[j]} | Prediction: {np.round(y_hat,4)} | W: {np.round(W, 4)}")
