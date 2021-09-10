# Sparse Coding

# Import modules
import numpy as np

# Read in vectors
X = np.array([
    [2.8, 1.2]]
, dtype = np.float32)

V = np.array([
    [4.0, 6.2, 2.1, -2.0],
    [1.3, 3.5, 0.5, 7.4]
])

y = np.array([
    1.9, 0, -2.0, 0
])

# y = np.array([
#     0.5, 0, 0.6, 0
# ])

# Compute reconstruction error
recon_error = np.linalg.norm(X - np.dot(V, y.T))

# COmpute sparsity
sparsity = len(y[y!=0])

cost = recon_error + sparsity

print("Sample:")
print(X)

print("\nSolution:")
print(y)

print(f"\nReconstruction error: {round(recon_error, 4)}")
print(f"Sparsity measure: {sparsity}")
print(f"Cost: {round(cost,4)}")
