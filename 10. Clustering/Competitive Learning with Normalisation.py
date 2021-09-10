# Competitive Learning

# Import modules
import numpy as np


# Data - AUGMENTED FORM
data = np.array([
    [1, -1,3],
    [1, 1,4],
    [1, 0,5],
    [1, 4,-1],
    [1, 3,0],
    [1, 5,1]
], dtype=float)

# Normalise data
for instance in range(len(data)):

    data[instance] = data[instance] - np.linalg.norm(data[instance])

centres = np.array([
    [0.5, -0.5, 1.5],
    [0.5, 0, 2.5],
    [0.5, 1.5, 0.0]
], dtype = float)

# Distance function
def distance(x_1, x_2):

    return np.linalg.norm(x_1 - x_2)


# Model params
nu = 0.1
c = centres.shape[0]
n = data.shape[0]

# Loop params
distances = np.empty(shape=(c, ))

patterns = data[[2, 0, 0, 4, 5]]

print("Centres at start:")
print(np.round(centres,4))


for pattern in range(patterns.shape[0]):

    print(f"\nPattern: {pattern + 1}")

    for centre in range(c):

        # Get distances
        distances[centre] = np.dot(centres[centre].T, patterns[pattern])
        
    # Classify the pattern
    j = np.argmax(distances)

    # Update weight param
    centres[j] += nu*(patterns[pattern])

    # Normalise weights
    centres[j] += centres[j] / np.linalg.norm(centres[j])

    print(f"Sample: {np.round(patterns[pattern],4)} | Distances: {np.round(distances,4)} | Clustter allocation: {j}")
    print("Centres after update:")
    print(np.round(centres,4))


# Classify instances
def assign_cluster(instances, centres):

    n = instances.shape[0]
    c = centres.shape[0]

    distances = np.empty(shape=(n, c))
    allocations = np.empty(shape=(n, ))

    for i in range(n):

        for centre in range(c):

            distances[i, centre] = distance(instances[i], centres[centre])

        # assign cluster
        allocations[i] = np.argmin(distances[i])

    return allocations

# Classify instances
allocations = assign_cluster(data, centres)
print("\nAllocation of clusters:")
print(allocations)

allocations = assign_cluster(np.array([[1, 0,-2]]), centres)
print("\nAllocation of clusters for new sample:")
print(allocations)