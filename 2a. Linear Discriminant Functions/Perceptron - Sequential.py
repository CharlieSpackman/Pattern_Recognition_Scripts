### Sequential Gradient Descent
import math
import numpy as np

# Input data - WITH BIAS
X = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [1, 0.5, 0.5],
    [1, 1, 0.5]
], dtype = np.float)

y = np.array([
    1,1,-1,-1
], dtype = np.float)

# Set weights - WITH BIAS
W = np.array([0.4, 1, 2], dtype = np.float)

# Set learning parameters
rate = 1.0
iterations = 2


### Perform sample normalisation ###
for i in range(len(X)):
    if y[i] < 0:
        X[i] = X[i] * -1

print("Before training")
print("W: {}".format(W))

# Start iterating
for epoch in range(iterations):

    print("\nIteration: {}".format(epoch+1))


    # Loop through all examples
    for j in range(len(X)):

        # Caculate prediction
        y_hat = np.dot(W, X[j].T)

        # Update weights
        if y_hat <= 0:
            W = W + rate * X[j]

        print(f"Sample: {X[j]} | Prediction: {y_hat} | W: {np.round(W, 4)}")

