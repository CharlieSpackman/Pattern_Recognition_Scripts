# Regulatory Feedback Network

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

# Define max selector
def max_selector(input_arr, epsilon):
    max_arr = np.max(input_arr)
    arr_dims = input_arr.shape

    if epsilon > max_arr:
        new_arr = np.ones(shape = arr_dims) * epsilon
        return new_arr
    
    else:
        return input_arr

# Update activations
iterations = 5
epsilon_1 = 0.01
epsilon_2 = 0.01

print(f"Initial Activations (y): {y}")

sum_of_rows = W.sum(axis=1)
norm_W = W / sum_of_rows[:, np.newaxis]

for j in range(iterations):
    
    # Update input units
    e = np.divide(X, max_selector(np.dot(W.T, y), epsilon_2))

    # Update output units
    y  = np.multiply(max_selector(y, epsilon_1), np.dot(norm_W, e))

    print(f"Iteration: {j + 1} | Error: {np.round(e, 4).tolist()} | Updated Activations: {np.round(y, 4).tolist()}")

