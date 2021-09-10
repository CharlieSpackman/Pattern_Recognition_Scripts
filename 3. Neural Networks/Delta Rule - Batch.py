# Delta Rule - Batch

# Import modules
import numpy as np


# Define inputs
X = np.array([
    [0],
    [1]
    ], dtype = np.float)

y = np.array([1,0], dtype = np.float)

theta = 1.5

W = np.array([2], dtype = np.float)


# Apply vector augmentation
ones = np.ones(shape=(len(X),1), dtype = np.float).reshape(-1,1)

# Add ones to X
X = np.concatenate((ones, X), axis=1)

# Add theta to W
W = np.append(-np.array([theta], dtype=np.float), W)


# Define Heaviside function
def heaviside(u):
    if u > 0:
        return 1
    elif u < 0:
        return 0
    else:
        return 0.5


# Apply sequential Delta Learning rule
iterations = 7
rate = 1

print("Initial W: {}".format(W))

for j in range(iterations):

    print(f"\nIteration: {j + 1}")
    batch_error = 0
    
    for i in range(len(X)):

        # Update W on instance
        net = np.dot(W, X[i])
        y_hat = heaviside(net)

        error = (y[i] - y_hat) * X[i]
        batch_error += error

        print(f"Sample: {X[i]} | Class (t): {int(y[i])} | net: {net} | H(net): {y_hat} | error: {error}")
    
    # Update W with batch error
    W = W + rate * batch_error
    print(f"Iteration {j + 1} complete | | Batch error: {batch_error} | W: {W} ")