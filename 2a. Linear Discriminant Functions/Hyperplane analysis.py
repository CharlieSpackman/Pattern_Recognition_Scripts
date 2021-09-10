# Hyperplane analysis

import numpy as np

# Weight vector - WITH BIAS
W = np.array([-5, 2, 1])

### Line Equation ###

# y = mX = c
m = -W[1] / W[2]
c = -W[0] / W[2]

print(f"Line equation: X2 = {round(m, 2)}X1 + {round(c, 2)}")


### Distance to hyperplane from origin ###

dist = abs(W[0]) / np.linalg.norm(W[1:])

print(f"\nDistance to Hyperplane from the origin: {round(dist, 4)}")


### Distance from point to the hyperplane ###

X = np.array([1,2,2])

g = np.dot(W.T, X)

point_dist = abs(g) / np.linalg.norm(W[1:])

print(f"\nDistance to from point {X} to Hyperplane: {round(point_dist, 4)}")
