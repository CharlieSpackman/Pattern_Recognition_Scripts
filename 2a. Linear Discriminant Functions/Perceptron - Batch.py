### Batch Gradient Descent
import math
import numpy as np

# Input data - WITH BIAS
X = np.array([
    [1, 1, 5],
    [1, 2, 5],
    [1, 4, 1],
    [1, 5, 1]
], dtype = np.float)

y = np.array([
    1,1,-1,-1
], dtype = np.float)

# Set weights - WITH BIAS
W = np.array([-25, 6, 3], dtype = np.float)

# Set learning parameters
rate = 1.0
iterations = 3


### Perform sample normalisation ###
for i in range(len(X)):
    if y[i] < 0:
        X[i] = X[i] * -1


print("Before training")
print("W: {}\n".format(W))

# Start iterating
for epoch in range(iterations):

    print("Iteration: {}".format(epoch+1))
    batch_error = np.array([0,0,0], dtype = np.float32)

    # Loop through all examples
    for j in range(len(X)):

        # Caculate prediction
        y_hat = np.dot(W, X[j].T)
        print(f"Sample: {X[j]} | Prediction: {y_hat} ")

        # Caculate error
        if y_hat <= 0:
            batch_error += X[j]

    # Update weights
    W = W + rate * batch_error

    print("Batch Error: {}".format(np.round(batch_error, 4)))
    print("W: {}\n".format(np.round(W, 4)))
