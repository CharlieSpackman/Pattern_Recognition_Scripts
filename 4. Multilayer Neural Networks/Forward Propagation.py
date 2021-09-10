# Forward Propagation

import numpy as np
import math

# Input data - NO BIAS
X = np.array([
    [0.3], 
    [-0.1]
    ], dtype = np.float32)

# Layer 1
W_ji = np.array([
    [-5, 2],
    [1,3],
    [3,-5]
    ], dtype = np.float32)

W_j0 = np.array([
    [5],
    [-1],
    [4]
    ])

# Layer 2
W_kj = np.array([
    [-1, 2, 4],
    [-5, 4, -1],
    ], dtype = np.float32)

W_k0 = np.array([
    [-4],
    [4]
    ])

# Define activaiton functions
def sym_sigmoid_activation(net):

    return (2 / (1 + math.exp(-2*net))) - 1
sym_sigmoid_activation_ = np.vectorize(sym_sigmoid_activation)

# def log_sigmoid_activation(net):

#     return 1 / (1 + math.exp(-1*net))
# log_sigmoid_activation_ = np.vectorize(log_sigmoid_activation)


# Add bias to input layer
X_ = np.concatenate((X, np.array([[1]])), axis = 0)

# Add bias vectors to Weights
W_ji = np.concatenate((W_ji, W_j0), axis=1)
W_kj = np.concatenate((W_kj, W_k0), axis=1)


# Calculate net at hidden layer
y = np.dot(W_ji, X_)
# Apply activation to hidden layer
y = sym_sigmoid_activation_(y)

# Add bias to y
y_ = np.concatenate((y, np.array([[1]])), axis = 0)

# Calculate net at output layer
z = np.dot(W_kj, y_)
# # Apply activation to output layer
# z = log_sigmoid_activation_(z)

# Print results

print(f"Instance: {X.tolist()} | Hidden layer output: {np.round(y,4).tolist()}| Output: {np.round(z,4).tolist()}")
