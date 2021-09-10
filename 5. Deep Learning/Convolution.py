import numpy as np

# Applying masks to produce feature mask


H = np.array(
    [
        [
            [-2.1, 8.9],
            [0.4, 7.1],
        ],
        [
            [-1.0, -6.5],
            [7.9, -2.9],
        ],
    ]
)


x = np.array(
    [
        [
            [0.2, 0.4, -0.1],
            [-0.8, -0.5, -1],
            [0.2, -1.0, 0.9],
        ],
        [
            [0.7, -0.4, 0.1],
            [0.6, -0.7, -0.5],
            [-0.8, -0.9, -0.6],
        ],
    ]
)


def strided_len(x_len, H_len, stride):
    return np.ceil((x_len - H_len + 1) / stride).astype(int)


def H_dilated_len(H_len, dilation):
    return (H_len - 1) * (dilation - 1) + H_len


def dilate_H(H, dilation):
    H_rows, H_cols = H[0].shape
    H_dilated = np.zeros((H.shape[0], H_dilated_len(H_rows, dilation), H_dilated_len(H_cols, dilation)))
    H_dilated[:, ::dilation, ::dilation] = H
    return H_dilated


def apply_mask(x, H, padding=0, stride=1, dilation=1):
    # x and H can have multiple channels in the 0th dimension
    if padding > 0:
        x = np.pad(x, pad_width=padding, mode='constant')[1:-1]

    if dilation > 1:
        H = dilate_H(H, dilation)

    H_rows, H_cols = H[0].shape
    x_rows, x_cols = x[0].shape

    fm_rows, fm_cols = strided_len(x_rows, H_rows, stride), strided_len(x_cols, H_cols, stride)
    feature_map = np.empty((fm_rows, fm_cols))
    for xf in range(fm_rows):
        for yf in range(fm_cols):
            xi, yi = xf * stride, yf * stride
            receptive_region = x[:, xi : xi + H_rows, yi : yi + H_cols]
            feature_map[xf, yf] = np.sum(H * receptive_region)
    return feature_map


# print(apply_mask(x, H))
# print(apply_mask(x, H, padding=1))
# print(apply_mask(x, H, padding=1, stride=2))
print(apply_mask(x, H, padding=1, stride=2, dilation=2))


# # 1x1 mask

# x = np.array(
#     [
#         [
#             [0.2, 0.4, -0.1],
#             [-0.8, -0.5, -1],
#             [0.2, -1.0, 0.9],
#         ],
#         [
#             [0.7, -0.4, 0.1],
#             [0.6, -0.7, -0.5],
#             [-0.8, -0.9, -0.6],
#         ],
#     ]
# )

# H = np.array([[[1]], [[-1]], [[0.5]]])
# print(apply_mask(x, H))
