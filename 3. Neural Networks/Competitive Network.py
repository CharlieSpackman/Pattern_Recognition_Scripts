# Comp netrowk

import numpy as np

X = np.array([
    [0.01,0.24,0.59, 0.50, 0.58]
])

W = np.array([
    [0.47,0.22,-0.85,0.07,-0.07],
    [0.09,0.44,0.62,-0.02,0.64],
    [-0.61,0.54,0.33,0.01,0.47],
    [0.35,0.55,0.32,0.67,-0.11]
])

y = np.dot(X, W.T)

print(np.round(y,4))