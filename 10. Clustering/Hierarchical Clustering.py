# Clustering

# Import modules
import numpy as np
from itertools import combinations
# Data
data = np.array([
    [-1,3],
    [1,2],
    [0,1],
    [4,0],
    [5,4],
    [3,2]
], dtype=float)


# Centers
centres = data.copy()

c = 3
n = data.shape[0]

# Distance function
def distance(x_1, x_2):

    return np.linalg.norm(x_1 - x_2)

def single_link(centres):

    c = centres.shape[0]
    combos = list(combinations(list(range(c)), 2))
    distances = np.empty(shape=(len(combos),))

    for i, combo in enumerate(combos):
        distances[i] = distance(centres[combo[0]], centres[combo[1]])

    # Pick min value
    j = np.argmin(distances)
    j_dist = np.min(distances)

    return combos[j], j_dist

# Loop 
for i in range(3):

    print("\n\nITERATION:", i+1)
    # find two closest 
    mergers, dist = single_link(centres)

    # Create new cluster for two closests points
    print("\nClusers being mergered:")
    print(mergers)
    
    print("\nNew cluster")
    new_cluster = np.mean((centres[mergers[0]], centres[mergers[1]]), axis = 0)
    print(new_cluster)

    print("\nDistance:", round(dist,4))

    print("\nClusters before merging")
    print(centres)
    # Remove old centres and add new one
    centres = np.delete(centres, mergers[1], axis=0)
    centres = np.delete(centres, mergers[0], axis=0)

    centres = np.concatenate((centres, [new_cluster]), axis=0)

    print("\nClusters after merging")
    print(centres)
