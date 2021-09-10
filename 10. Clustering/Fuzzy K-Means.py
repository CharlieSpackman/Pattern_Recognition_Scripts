# Clustering

# Import modules
import numpy as np
import math
np.set_printoptions(suppress=True)

# Data
data = np.array([
    [-1,3],
    [1,4],
    [0,5],
    [4,-1],
    [3,0],
    [5,1]
], dtype=np.float32)

weights = np.array([
    [1,0],
    [0.5,0.5],
    [0.5,0.5],
    [0.5,0.5],
    [0.5,0.5],
    [0,1]
], dtype=np.float32)

# Distance function
def distance(x_1, x_2):

    return np.linalg.norm(x_1 - x_2)

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

# Params
c = 2
n = data.shape[0]
b = 2.0


# Create initial cluster centres
centres = np.concatenate(([np.divide(np.dot(np.power(weights[:, 0], b), data), np.power(weights[:,0],2).sum())], [np.divide(np.dot(np.power(weights[:, 1], b), data), np.power(weights[:,1],2).sum())]), axis=0)

for j in range(n):

    weights[j, 0] = (1.0 / math.pow(distance(data[j], centres[0]), (2.0 / (b-1.0)))) / ( math.pow((1.0 / distance(data[j], centres[0])) , (2.0 / (b-1.0))) + math.pow((1.0 / distance(data[j], centres[1])) , (2.0 / (b-1.0))))
    weights[j, 1] = (1.0 / math.pow(distance(data[j], centres[1]), (2.0 / (b-1.0)))) / ( math.pow((1.0 / distance(data[j], centres[0])) , (2.0 / (b-1.0))) + math.pow((1.0 / distance(data[j], centres[1])) , (2.0 / (b-1.0))))

print("\nCentres after iteration 1")
print(np.round(centres,4))
print("Membership after iteration 1")
print(np.round(weights,5))
print("Allocations after iteration 1")
print(assign_cluster(data, centres))


complete = False
iters = 0
while complete == False:

    centres_old = centres
    
    centres = np.concatenate(([np.divide(np.dot(np.power(weights[:, 0], b), data), np.power(weights[:,0],2).sum())], [np.divide(np.dot(np.power(weights[:, 1], b), data), np.power(weights[:,1],2).sum())]), axis=0)

    for j in range(n):

        weights[j, 0] = (1.0 / math.pow(distance(data[j], centres[0]), (2.0 / (b-1.0)))) / ( math.pow((1.0 / distance(data[j], centres[0])) , (2.0 / (b-1.0))) + math.pow((1.0 / distance(data[j], centres[1])) , (2.0 / (b-1.0))))
        weights[j, 1] = (1.0 / math.pow(distance(data[j], centres[1]), (2.0 / (b-1.0)))) / ( math.pow((1.0 / distance(data[j], centres[0])) , (2.0 / (b-1.0))) + math.pow((1.0 / distance(data[j], centres[1])) , (2.0 / (b-1.0))))

    print("\nCentres after iteration {}:".format(iters+2))
    print(np.round(centres,4))
    print("Membership after iteration {}:".format(iters+2))
    print(np.round(weights,5))
    print("Allocations after iteration {}:".format(iters+2))
    print(assign_cluster(data, centres))

    iters += 1

    if np.all(np.abs(centres - centres_old) < 0.5):
        complete = True



