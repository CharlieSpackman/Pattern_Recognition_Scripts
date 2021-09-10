# GAN Forward Pass

import numpy as np
import math

# Data
X = np.array([
    [1,2],
    [3,4],
    [5,6],
    [7,8]
], dtype = np.float32)

# Classes: 1 = Real, 0 = Fake
classes = np.array([
    1,1,0,0
])

# Model weights
theta = np.array([
    0.1, 0.2
], dtype = np.float32)

# Define Discriminator
def D(X, theta):

    x_1 = X[0]
    x_2 = X[1]

    theta_1 = theta[0]
    theta_2 = theta[1]

    return 1 / (1 + math.exp(-1 * (theta_1*x_1 - theta_2*x_2 - 2)))

# Define cost function
def cost(X, classes, theta):
    
    # Real == 1, Fake == 0
    real_instances = X[classes == 1]
    fake_instances = X[classes == 0]

    real_count = real_instances.shape[0]
    fake_count = fake_instances.shape[0]

    # Calculate Discriminator loss
    discriminator_loss = 0
    print("\nCaculate loss on the Discriminator")
    for instance in range(real_count):
        
        instance_loss = (1 / real_count) * math.log(D(real_instances[instance], theta))
        discriminator_loss += instance_loss

        print(f"Sample: {real_instances[instance]} | Loss: {round(instance_loss,4)}")

    print(f"Total loss: {round(discriminator_loss,4)}")

    # Calculate Generator loss
    generator_loss = 0
    print("\nCaculate loss on the Generator")
    for instance in range(fake_count):

        instance_loss = (1 / fake_count) * math.log(1 - D(fake_instances[instance], theta))
        generator_loss += instance_loss

        print(f"Sample: {fake_instances[instance]} | Loss: {round(instance_loss,4)}")

    print(f"Total loss: {round(generator_loss,4)}")

    return discriminator_loss + generator_loss


# Print loss
print(f"Forward Pass using theta={theta} and the following data:")
print(X)
print(f"Classes (1==Real, 0==Fake): {classes}")

loss = cost(X, classes, theta)
print(f"\nTotal loss: {round(loss,4)}")
