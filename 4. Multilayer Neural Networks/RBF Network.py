# RBF Network

import numpy as np
import math
import itertools


### INPUT DATA ###

# Input data - NO BIAS
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
    ], dtype = np.float32)

# Layer 1 - CENTRES - ADD HERE
c_ji = np.array([
    [0.0, 0.0],
    [1.0, 1.0]
    ], dtype = np.float32)


# Layer 2 - NO BIAS
W_kj = np.array([
    [0.0],
    [0.0]
    ], dtype = np.float32)

W_k0 = np.array([
    [0.0]
    ])

target = np.array([
    [0],
    [1],
    [1],
    [0]
    ], dtype = np.float32)

### Define activaiton functions ###

def guassain_RBF(d, sigma):

    return math.exp(- math.pow(d,2) / (2 * math.pow(sigma,2)))
# guassain_RBF_ = np.vectorize(guassain_RBF)


# def sym_sigmoid_activation(net):

#     return (2 / (1 + math.exp(-2*net))) - 1

# sym_sigmoid_activation_ = np.vectorize(sym_sigmoid_activation)

# def sym_sigmoid_derivative(net):

#     numerator = 4 * math.exp(-2 * net)
#     denominator = math.pow(1 + math.exp(-2 * net), 2)
#     return numerator / denominator

# sym_sigmoid_derivative_ = np.vectorize(sym_sigmoid_derivative)

# def log_sigmoid_activation(net):

#     return 1 / (1 + math.exp(-1*net))
# log_sigmoid_activation_ = np.vectorize(log_sigmoid_activation)


### Define cost function ###
def cost(target, z):

    return np.linalg.norm(target - z) * 0.5


### Add Biases ###

# Add bias vectors to Weights
W_kj = np.concatenate((W_kj, W_k0), axis=0)


### Determine sigma
hidden_layers = c_ji.shape[0]
combos = list(itertools.combinations(list(range(c_ji.shape[0])), 2))

distances = np.zeros(shape = (len(combos),1))
for id, item in enumerate(combos):
    dist = np.linalg.norm(c_ji[item[0]] - c_ji[item[1]])
    distances[id] = dist

p_max = np.max(distances)
p_avg = np.sum(distances) / distances.shape[0]

sigma_max = p_max / math.sqrt(2*c_ji.shape[0])
sigma_avg = p_avg * 2

print(f"\nSigma MAX is being used: {sigma_max}\n")

# Calculate hidden layer output
y = []

# Loop over instances
for instance in range(len(X)):
    
    y_j = []

    # Loop over centres
    for centre in range(len(c_ji)):

        # Calculate activation
        d = np.linalg.norm(X[instance] - c_ji[centre])
        y_j.append(round(guassain_RBF(d, sigma_max),4))

    y.append(y_j)

print("Activations at hidden layer for each instance")
print(y)


# Determine Output weights by method of least squares
# Convert activations to matrix and add bias
y = np.array(y)
y = np.concatenate((y, np.ones(shape=(X.shape[0])).reshape(-1,1)), axis = 1)


# Check if y is square
if y.shape[0] == y.shape[1]:
    y_inv = np.linalg.inv(y)

    for k in range(len(W_kj)):

        W_kj = np.round(np.dot(y_inv, target),4)

else:
    y_non_sqr_inv = np.dot(np.linalg.inv(np.dot(y.T, y)), y.T)

    # for k in range(len(W_kj)):

    W_kj = np.round(np.dot(y_non_sqr_inv, target),4)

print("\nHidden -> Output Weights")
print(W_kj)

# Complete a forward pass
z = np.round(np.dot(y, W_kj),4)
print("\nForward pass")
print(z)




### Forward pass based on new data
print("\nOutputs for unseen data")
X = np.array([
    [0.5, -0.1],
    [-0.2, 1.2],
    [0.8, 0.3],
    [1.8, 0.6]
    ], dtype = np.float32)

y = []

# Loop over instances
for instance in range(len(X)):
    
    y_j = []

    # Loop over centres
    for centre in range(len(c_ji)):

        # Calculate activation
        d = np.linalg.norm(X[instance] - c_ji[centre])
        y_j.append(round(guassain_RBF(d, sigma_max),4))

    y.append(y_j)

print("Activations at hidden layer for each instance")
print(y)

y = np.array(y)
y = np.concatenate((y, np.ones(shape=(X.shape[0])).reshape(-1,1)), axis = 1)

# Complete a forward pass
z = np.round(np.dot(y, W_kj),4)
print("\nForward pass")
print(z)
