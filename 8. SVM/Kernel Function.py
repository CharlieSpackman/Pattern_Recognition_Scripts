# Kernel Function

import numpy as np

X = np.array([
    [2,2],
    [2,-2],
    [-2,-2],
    [-2,2],
    [1,1],
    [1,-1],
    [-1,-1],
    [-1,1]
])

def kernel(X):

    X_new = np.empty(shape=X.shape)

    for instance in range(len(X)):

        x_1 = X[instance][0]
        x_2 = X[instance][1]

        mag = np.linalg.norm(X[instance])
        if mag > 2:

            X_new[instance] = np.array([
                4 - (x_2/2) + abs(x_1 - x_2),
                4 - (x_1/2) + abs(x_1 - x_2)
            ])

        
        else:

            X_new[instance] = np.array([
                x_1-2,
                x_2-3
            ])

    return X_new
            

print(np.round(kernel(X),4))

