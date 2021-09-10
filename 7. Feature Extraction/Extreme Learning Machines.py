# Extreme Learning Machines

# Import modules
import numpy as np

# Read in vectors
# X = np.array([
#     [1,2],
#     [2,1],
#     [3,3],
#     [6,5],
#     [7,8]]
# , dtype = np.float32)

V = np.array([
    [0.9, 0.3, 0, 0.5],
    [0.1, -0.8,0.7, -0.1],
    [0.3, 0.2, -0.6, -0.4],
    [-1, 0.2, 1, 0],
    [0.3, -0.5, -0.8, -0.9]
])

W = np.array([
    [0.15, 0.15, 0.25, -0.1, 0.25, -0.1],
    [0.1, 0.1, -0.05, 0.15, -0.05, 0.15]
], dtype = np.float32)

X = np.array([
    [1,0.5,0.9,0.4],
    [1,0.9,0,-0.5]])

def heavisde(net):

    if net >= 0:
        return 1
    else:
        return 0
heavisde = np.vectorize(heavisde)


# Caclulate y - hidden layer output
y = np.dot(V, X.T)
y = heavisde(y)
y_ = np.concatenate((np.array([np.zeros(shape=y.shape[1])]), y), axis = 0)


# Calculate z = output layer
z = np.dot(W, y_)

# Print results
print("Samples:")
print(X)

print("\nOutput from hidden layer (y):")
print(y)

print("\n Output from final layer:")
print(z)