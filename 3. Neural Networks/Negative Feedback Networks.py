# Negative Feedback Networks

# Import modules
import numpy as np


# Define inputs
X = np.array([
    [1],
    [1],
    [0]
    ], dtype = np.float)

y = np.array([
    [0],
    [0]
    ], dtype = np.float)

W = np.array([
    [1,1,0],
    [1,1,1]
    ], dtype = np.float)


# Update activations
iterations = 5
alpha = 0.25

print(f"Initial Activations (y): {y}")

for j in range(iterations):
    
    # Update input units
    e = X - np.dot(W.T, y)

    # Update output units
    y  = y + alpha * np.dot(W, e)

    print(f"Iteration: {j + 1} | Error: {np.round(e, 4).tolist()} | Updated Activations: {np.round(y, 4).tolist()}")
