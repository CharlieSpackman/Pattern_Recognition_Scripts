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

def sgn(x):
    if x > 0:
        return 1
    else:
        return -1

# Set weights - WITH BIAS
W = np.array([0,0], dtype = np.float)

# Set learning parameters
rate = 0.01
iterations = 2

print("Before training")
print("W: {}\n".format(W))

# Start iterating
for epoch in range(iterations):

    batch_error = np.array([0,0], dtype = np.float32)

    # Loop through all examples
    for j in range(len(X)):

        # Caculate prediction
        y_hat = sgn(np.dot(W, X[j].T))

        # Caculate error
        if y_hat != y[j]:
            batch_error += X[j] * y[j]

    # Update weights
    W = W + rate * batch_error

    print("Iteration: {}".format(epoch+1))
    print("Batch Error: {}".format(np.round(batch_error, 4)))
    print("W: {}\n".format(np.round(W, 4)))
