# Clustering

# Import modules
import numpy as np

# Data
data = np.array([
    [-1,3],
    [1,4],
    [0,5],
    [4,-1],
    [3,0],
    [5,1]
], dtype=float)


# Centers
centres = np.array([
    [-1,3],
    [5,1]
], dtype=float)


# Distance function
def distance(x_1, x_2):

    return np.linalg.norm(x_1 - x_2)

# Params
ITERATIONS = 10
c = centres.shape[0]
n = data.shape[0]

# Loop objects
distances = np.empty(shape=(n, c))
centre_allocation = np.empty(shape=(n, ))

print("Centres before updating:")
print(centres)

# Loop
for iters in range(ITERATIONS):

    print(f"\nIteration: {iters+1}")

    ### Compute cluster allocation
    for i in range(n):
        
        # Get distance from centres
        for centre in range(c):

            distances[i, centre] = distance(data[i], centres[centre])

    
    # Assign clusters
    for i in range(n):

        centre_allocation[i] = np.argmin(distances[i])

        print(f"Sample: {data[i]} | Distances to centres: {np.round(distances[i],4)} | Assignend to: {centre_allocation[i]}")
    
    ### Recompute centres
    for centre in range(c):

        if np.sum(centre_allocation==centre) != 0:
            centres[centre] = data[centre_allocation==centre].mean(axis=0)

    ### print results
    print(f"Centres after iteration:")
    print(centres)







