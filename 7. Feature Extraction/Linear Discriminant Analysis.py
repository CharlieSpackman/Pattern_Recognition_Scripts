# Linear Discriminant Analysis

# Import modules
import numpy as np
import pandas as pd
from itertools import combinations
import math

# Read in vectors
X = np.array([
    [1,2],
    [2,1],
    [3,3],
    [6,5],
    [7,8]]
)

y = np.array(
    [1,1,1,2,2]
)

classes = np.unique(y).tolist()

W = np.array([2,-3])

sb = 0
sw = 0

### Calculate between class scatter
df = pd.DataFrame(X)
df[2] = y

df_mean = df.groupby([2]).mean()
centres = df_mean.to_numpy()

combos = list(combinations(list(range(len(centres))), 2))

for combo in combos:

    sb += math.pow(np.dot(W, centres[combo[0]] - centres[combo[1]]),2)

# Calculate sw
s = np.zeros(shape=(len(centres)))

for centre in range(len(centres)):

    for instance in range(len(X)):
        
        if y[instance] == classes[centre]:
            s[centre] += math.pow(np.dot(W, (X[instance] - centres[centre]).T),2)

sw = s.sum()

# Calculate cost
j = sb / sw

print(f"centres: {centres.tolist()} | sb: {sb} | sw: {sw} | j(W): {round(j,4)}")