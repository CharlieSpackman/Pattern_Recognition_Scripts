# Karhunen-Loeve Transform

# Import modules
import numpy as np
import math

# Read in vectors
X = np.array(
    [[1,2,1],
    [2,3,1],
    [3,5,1],
    [2,2,1]]
)

PCs = 2

# Calculate mean 
mu = X.mean(axis=0)

# Calculate covariance matrix
C = np.cov(X, rowvar = False, bias = True)

# Get eigenvalues and eigenvectors of C
evals, evec = np.linalg.eig(C)


# Reorder eigenvectors and get first two elements
evec = np.flip(evec[:, evals.argsort()],axis=1)
evec = evec[:,:PCs]

# Reorder eigenvalues and get first two elements
evals = np.flip(evals[evals.argsort()])
evals = evals[:PCs]


# Calculate y
y = np.round(np.dot((X - mu), evec),4)

print("Sample:")
print(X)

print("\nEigenVector:")
print(evec)

print(f"\nEigenValues: {evals}")

print("\nProjected vector (y):")
print(y)