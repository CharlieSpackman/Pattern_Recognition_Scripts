# Activation functions

import numpy as np
import math

# Define inputs
X = np.array([
    [1, 0.5, 0.2],
    [-1, -0.5, -0.2],
    [0.1,-0.1,0]
], dtype = np.double)

# Define activation functions
def relu(net):
    if net >= 0:
        return net

    else:
        return 0
relu = np.vectorize(relu)

def lrelu(net, a):
    if net >= 0:
        return net

    else:
        return a * net
lrelu = np.vectorize(lrelu)

def prelu(net, a):
    if net >= 0:
        return net

    else:
        return a * net
prelu = np.vectorize(prelu) 

def sgn(net):
    if net >= 0:
        return 1

    else:
        return -1
sgn = np.vectorize(sgn)

def heaviside(net):
    threshold = 0.1
    if abs(net - threshold) == 0.0:
        return 0.5
    elif (net - threshold) > 0.0:
        return 1.0
    else:
        return 0.0
heaviside = np.vectorize(heaviside)

def sys_sigmoid(net):
    
    return (2 / (1 + math.exp(-2 * net))) - 1
sys_sigmoid = np.vectorize(sys_sigmoid)


def log_sigmoid(net):

    return 1 / (1 + math.exp(-1 * net))
log_sigmoid = np.vectorize(log_sigmoid)


def rbf(net):

    return math.exp(-1 * math,pow(net,2)) 
rbf = np.vectorize(rbf)



def tanh_tranf(net):
    return math.tanh(net)
tanh_tranf = np.vectorize(tanh_tranf)


print(relu(X))

print(lrelu(X, a=0.1))

print(tanh_tranf(X))

print(heaviside(X))