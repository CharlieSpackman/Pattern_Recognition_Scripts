# Batch Normalisation

import numpy as np
import math
import statistics

from numpy.core.fromnumeric import mean

# Define inputs
X = np.array([
    [[1, 0.5, 0.2],
    [-1, -0.5, -0.2],
    [0.1,-0.1,0]],
    
    [[1, -1, 0.1],
    [0.5, -0.5, -0.1],
    [0.2, -0.2, 0]],

    [[0.5, -0.5, -0.1],
    [0, -0.4, 0],
    [0.5, 0.5, 0.2]],

    [[0.2, 1, -0.2],
    [-1, -0.6, -0.1],
    [0.1, 0, 0.1]]]
    )

def variance_(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return mean, variance 


def batch_norm(input_arrs, beta, gamma, epsilon):

    result = np.zeros(shape=input_arrs.shape)

    for i in range(input_arrs.shape[1]):

        for j in range(input_arrs.shape[2]):

            batch = []

            for k in range(input_arrs.shape[0]):

                batch.append(input_arrs[k, i, j])

            expectation, variance = variance_(batch)


            for k in range(input_arrs.shape[0]): 

                result[k,i,j] = beta + gamma * (input_arrs[k,i,j] - expectation) / math.sqrt(variance + epsilon)

    return result

result = batch_norm(X, 0, 1, 0.1)

print(f"Sample:")
print(X)

print(f"\nResult:")
print(np.round(result,4))
