### Batch Gradient Descent
import math
import numpy as np

# Input data - WITH BIAS
X = np.array([
    [1,1.5],
    [1,3.5],
    [1,3],
    [1,5],
    [1,2]
], dtype = np.float)

y = np.array([
    1,
    3,
    2,
    3,
    2.5
], dtype = np.float)

# Set weights - WITH BIAS
W = np.array([0,0], dtype = np.float)

# Set margin 
b = np.array([
    1,
    3,
    2,
    3,
    2.5
], dtype = np.float)

# Set learning parameters
rate = 0.01
iterations = 2


### Perform sample normalisation ###
for i in range(len(X)):
    if y[i] < 0:
        X[i] = X[i] * -1


print("Before training")
print("W: {}\n".format(W))

# Start iterating
for epoch in range(iterations):

    batch_error = np.array([0,0], dtype = np.float32)

    # Loop through all examples
    for j in range(len(X)):

        # Caculate prediction
        y_hat = np.dot(W, X[j].T)

        # Caculate error
        batch_error += X[j] * (b[j] - y_hat)

    # Update weights
    W = W + rate * batch_error

    print("Iteration: {}".format(epoch+1))
    print("Batch Error: {}".format(np.round(batch_error, 4)))
    print("W: {}\n".format(np.round(W, 4)))
