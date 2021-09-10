### Multiclass Sequential Gradient Descent
import math
import numpy as np

# Input data - WITH BIAS
X = np.array([
    [1, 1,1],
    [1, 2,0],
    [1, 0,2],
    [1, -1,1],
    [1, -1,-1]
], dtype = np.float)

y = np.array([
    1,1,2,2,3
], dtype = np.float)

# Set weights - WITH BIAS
W = np.array([
    [0, 0, 0],
    [0,0,0],
    [0,0,0]
    ], dtype = np.float)

# Set learning parameters
rate = 1.0
iterations = 3

print("Before training")
print("W_1: {}".format(W[0]))
print("W_2: {}".format(W[1]))
print("W_3: {}".format(W[2]))

# Start iterating
for epoch in range(iterations):

    print("\nIteration: {}".format(epoch+1))

    # Loop through all examples
    for j in range(len(X)):

        # Caculate prediction
        y_hat = np.array([np.dot(W[0], X[j].T), np.dot(W[1], X[j].T), np.dot(W[2], X[j].T)]) 

        max_val = np.max(y_hat)
        max_vals_list = [1 for item in y_hat.tolist() if item == max_val]
        if sum(max_vals_list) > 1:
            if max_vals_list[2] == 1:
                c = 2
            elif max_vals_list[1] == 1:
                c = 1
            else:
                c = 0
        
        else:
            c = np.argmax(y_hat)

        # Update weights if required
        if c != (y[j] - 1):

            id = int(y[j]) - 1
            W[id] = W[id] + rate * X[j]
            W[c] = W[c] - rate * X[j]

        print(f"Sample: {X[j]} | Prediction: {y_hat} | c: {c+1}| | Class: {int(y[j])} | W: {np.round(W[0], 4)}, {np.round(W[1], 4)}, {np.round(W[2], 4)}")

