# Competitive Learning

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

centres = np.array([
    [0,5]
], dtype = float)

patterns = np.array([
    [-1,3],
    [-1,3],
    [3,0],
    [5,1]
], dtype = np.float32)

# Distance function
def distance(x_1, x_2):

    return np.linalg.norm(x_1 - x_2)

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


# Model params
nu = 0.5
c = centres.shape[0]
n = data.shape[0]
theta = 3

# Loop params
distances = np.empty(shape=(c, ))

print("Centres before learning:")
print(centres)

for pattern in range(patterns.shape[0]):

    print(f"\nIteration: {pattern + 1}")

    # Find neatest cluster
    for centre in range(distances.shape[0]):
        distances[centre] = distance(patterns[pattern], centres[centre])

    j = np.argmin(distances)
    j_dist = np.min(distances)

    print(f"Sample: {patterns[pattern]} | Distances: {np.round(distances,4)}")

    if j_dist < theta:
        print(f"Min dist: {j_dist} | Update centre for: {j}")
        centres[j] += nu*(patterns[pattern]-centres[j])
    else:
        print(f"Adding new centre")
        centres = np.concatenate((centres, np.array([patterns[pattern]])),axis=0)
        distances = np.concatenate((distances, np.array([0])),axis=0)

    # Update distance 
    print("Centres after iteration")
    print(centres)

allocations = assign_cluster(data, centres)
print("\nAllocation of clusters:")
print(allocations)


allocations = assign_cluster(np.array([[0,-2]]), centres)
print("\nAllocation of clusters:")
print(allocations)