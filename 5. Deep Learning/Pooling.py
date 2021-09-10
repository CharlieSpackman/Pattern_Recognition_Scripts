import numpy as np


x = np.array(
    [
        [
            [0.2, 1, 0, 0.4],
            [-1, 0, -0.1, -0.1],
            [0.1, 0, -1, -0.5],
            [0.4, -0.7, -0.5, 1],
        ]
    ]
)
    

def strided_len(x_len, H_len, stride):
    return np.ceil((x_len - H_len + 1) / stride).astype(int)


def pooling(x, pooling_function, region=(2, 2), stride=1):
    x_rows, x_cols = x[0].shape
    pool_rows, pool_cols = strided_len(x_rows, region[0], stride), strided_len(x_cols, region[1], stride)
    pool = np.empty((x.shape[0], pool_rows, pool_cols))
    for xp in range(pool_rows):
        for yp in range(pool_cols):
            xi, yi = xp * stride, yp * stride
            pooling_region = x[:, xi : xi + region[0], yi : yi + region[1]]
            pool[:, xp, yp] = pooling_function(pooling_region)
    return pool


print(pooling(x, np.mean, (2, 2), stride=2))
print(pooling(x, np.max, (2, 2), stride=2))
print(pooling(x, np.max, (3, 3), stride=1))