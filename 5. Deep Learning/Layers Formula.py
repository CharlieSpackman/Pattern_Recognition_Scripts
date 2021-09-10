# Layers Formula

# Define layer functions
def convolution(input_shape=(200,200,3), mask_shape = (40,40,3), mask_count=40, stride=1, padding=0):

    width = input_shape[0]
    height = input_shape[1]
    channels = input_shape[2]

    mask_width = mask_shape[0]
    mask_height = mask_shape[1]
    mask_channels = mask_shape[2]

    new_width = int(1 + (width - mask_width + 2*padding) / stride)
    new_height = int(1 + (height - mask_height + 2*padding) / stride)

    if channels != mask_channels:

        print("Channels not aligned!")

    return (new_width, new_height, mask_count)


def pooling(input_shape=(200,200,3), pooling_mask_shape = (40,40), stride=1, padding=0):

    width = input_shape[0]
    height = input_shape[1]
    channels = input_shape[2]

    pooling_mask_width = pooling_mask_shape[0]
    pooling_mask_height = pooling_mask_shape[1]

    new_width = int(1 + (width - pooling_mask_width + 2*padding) / stride)
    new_height = int(1 + (height - pooling_mask_height + 2*padding) / stride)


    return (new_width, new_height, channels)

def one_by_one(input_shape=(100,100,80), mask_count = 20):

    return (input_shape[0], input_shape[1], mask_count)


# Input shape
input_shape = (339,339,3)


# Add layers as needed
layer_1 = convolution(input_shape, (13,13,3), 50, stride=3, padding = 5)

layer_2 = pooling(layer_1, (5,5), stride =2)

layer_3 = one_by_one(layer_2, 42)


# Print results
print(f"Input: {input_shape}")
print(f"Layer 1: {layer_1}")
print(f"Layer 2: {layer_2}")
print(f"Layer 3: {layer_3}")