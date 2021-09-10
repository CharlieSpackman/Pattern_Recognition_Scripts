### Sequential Gradient Descent
import math
import numpy as np

# Input data - with Bias
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
W = np.array([0.4,1,2], dtype = np.float)

def sgn(x):
    if x > 0:
        return 1
    else:
        return -1


# Set learning parameters
rate = 1.0
iterations = 3

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
        if sgn(y_hat) != y[j]:
            W = W + rate * X[j] * y[j]

        print(f"Sample: {X[j]} | Prediction: {y_hat} | W: {np.round(W, 4)}")
