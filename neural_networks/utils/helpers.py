import numpy as np
###### =======LOSSES======= #####
def cross_entropy_loss(ytrue, yhat):
    return -np.sum(ytrue * np.log(yhat))

def sse_loss(ytrue, yhat):
    return 0.5 * np.sum(np.square(y_hat - y))


###### =======ACTIVATIONS======= #####
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(-1, 1)

def identity(z):
    return z

def relu(z):
    return np.where(z >= 0 , z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


###### =======GRADIENTS======= #####
def grad_sigmoid(z):
    return sigmoid(z) * (1- sigmoid(z))

def grad_relu(z):
    return np.where(z >= 0, 1, 0)

def grad_softmax(ytrue, yhat):
    return ytrue - yhat

def grad_sse(X, w, y):
    return X.T @ (X@w - y)