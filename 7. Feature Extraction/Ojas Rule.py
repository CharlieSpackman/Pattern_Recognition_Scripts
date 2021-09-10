# Ojas rule

import numpy as np

X = np.array([
    [0.0, 1.0],
    [3.0, 5.0],
    [5.0, 4.0],
    [5.0, 6.0],
    [8.0, 7.0],
    [9.0, 7.0]
])

PCs = 2

# Calculate mean 
mu = X.mean(axis=0)
X = X - mu

W = np.array([-1.0,0.0])

rate = 0.01
iterations = 2
error = 0

print(f"Initial W: {W}\n")

for epoch in range(iterations):

    print(f"Iteration: {epoch+1}")

    batch_error = 0

    for instance in range(len(X)):

        # Forward pass
        y = np.dot(X[instance], W.T)

        batch_error += y * (X[instance] - y * W)

        print(f"Sample: {X[instance]} | y = WX: {y} | batch error (cum): {np.round(batch_error,4)}")

    W += rate * batch_error
    print(f"Weight update after batch: {np.round(W,4)}\n")