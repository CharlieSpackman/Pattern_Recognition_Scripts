# BackPropagation

import numpy as np
import math

### INPUT DATA ###

# Input data - NO BIAS
X = np.array([
    [0.1], 
    [0.9]
    ], dtype = np.float32)

# Layer 1 - NO BIAS
W_ji = np.array([
    [0.5, 0.3],
    [0, -0.7]
    ], dtype = np.float32)

W_j0 = np.array([
    [0.2, 0]
    ])

# Layer 2 - NO BIAS
W_kj = np.array([
    [0.8],
    [1.6]
    ], dtype = np.float32)

W_k0 = np.array([
    [-0.4]
    ])

target = np.array([
    [0.5]
    ])

### Define activaiton functions ###
def sym_sigmoid_activation(net):

    return (2 / (1 + math.exp(-2*net))) - 1

sym_sigmoid_activation_ = np.vectorize(sym_sigmoid_activation)

def sym_sigmoid_derivative(net):

    numerator = 4 * math.exp(-2 * net)
    denominator = math.pow(1 + math.exp(-2 * net), 2)
    return numerator / denominator

sym_sigmoid_derivative_ = np.vectorize(sym_sigmoid_derivative)

# def log_sigmoid_activation(net):

#     return 1 / (1 + math.exp(-1*net))
# log_sigmoid_activation_ = np.vectorize(log_sigmoid_activation)


### Define cost function ###
def cost(target, z):
    
    return np.linalg.norm(target - z) * 0.5


### Add Biases ###

# Add bias to input layer
X_ = np.concatenate((X, np.array([[1]])), axis = 0)

# Add bias vectors to Weights
W_ji = np.concatenate((W_ji, W_j0), axis=0)
W_kj = np.concatenate((W_kj, W_k0), axis=0)


### Model parameters ###
rate = 0.25
iterations = 1


### Feedforward operation ###
print("---FEEDFORWARD OPERATION---")
print(f"Forward Propagation of sample: {np.round(X,4).tolist()}")
# Calculate net at hidden layer
net_y = np.dot(W_ji.T, X_)
# Apply activation to hidden layer
y = sym_sigmoid_activation_(net_y)

print(f"Hidden Layer | net_y: {np.round(net_y,4).tolist()} | y: {np.round(y,4).tolist()}")

# Add bias to y
y_ = np.concatenate((y, np.array([[1]])), axis = 0)
# net_y_ = np.concatenate((net_y, np.array([[1]])), axis = 0)


# Calculate net at output layer
net_z = np.dot(W_kj.T, y_)
# Apply activation to output layer
z = sym_sigmoid_activation_(net_z)

print(f"Output Layer | net_z: {np.round(net_z,4).tolist()} | z: {np.round(z,4).tolist()}\n")


### BackProp operation ###
print("---BACKPROP OPERATION---")

# Update weights from hidden to output layer
delta_z = (target - z) * sym_sigmoid_derivative_(net_z)
W_kj = W_kj + rate * np.dot(y_, delta_z)

print(f"Hidden -> Output Layer | delta_z: {np.round(delta_z,4).tolist()} | W_kj: {np.round(W_kj,4).tolist()}")

# Update weights from input to hidden layer
delta_y = sym_sigmoid_derivative_(net_y) * np.dot(W_kj[:-1], delta_z)
W_ji = W_ji + rate * np.dot(X_, delta_y.T)

print(f"Input -> Hidden Layer | delta_y: {np.round(delta_y,4).tolist()} | W_ji: {np.round(W_ji,4).tolist()}\n")


