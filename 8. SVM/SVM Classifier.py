# SVM

import numpy as np 

# Define linear equation function
def convert_to_linear(W_0, W_1, W_2):

    # y = mX = c
    m = -W_1 / W_2
    c = -W_0 / W_2

    print(f"Line equation: X2 = {np.round(m, 2)}X1 + {np.round(c, 2)}")

# Support vectors
SVs = np.array([
    [9, 3, 1.7321], 
    [144, 12, 3.4641],
    [81, 9, 3]
], dtype = np.float32)

count_SVs = len(SVs)

classes = np.array([1,1,-1])
# Define simultaneous equations
equations = []

# Get lambda coefficients
for sv in range(len(SVs)):
    equations.append(np.dot(SVs[sv], SVs.T) * classes)

# Add final constraint
equations.append(classes)

# Add the equals cofficients
equals = np.array([1, 1, 1, 0]).reshape(-1,1)

equations = np.concatenate((equations, equals), axis = 1)

# Solve for these cofficients
targets = np.append(classes, np.array([[0]])).reshape(-1,1)


# Solve equations
lambdas = np.dot(np.linalg.inv(equations), targets)


# Get Hyperplane equation W
W = np.dot(lambdas[:-1].T, SVs*classes.reshape(-1,1))
W_0 = lambdas[-1]

# Get margin
margin = 2 / np.linalg.norm(W)


# Print results
print("Support Vectors:")
print(SVs)

print("\nClasses:")
print(classes)

print("\nLambda equations:")
print(equations)

print("\nLambda values:")
print(lambdas[:-1])

print("\nHyperplane equation W:")
print(np.round(W,4))
print(W_0)

print(f"Margin: {np.round(margin,4)}")

print("\nLinear Equation:")
convert_to_linear(W[0][0], W[0][1], W_0)

