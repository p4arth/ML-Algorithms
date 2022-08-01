import sys
import os
from pathlib import Path
import numpy as np
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass

from neural_networks.utils.helpers import cross_entropy_loss, sse_loss
from neural_networks.utils.helpers import softmax, identity, relu, sigmoid
from neural_networks.utils.helpers import grad_sigmoid, grad_relu, grad_softmax, grad_sse

class NeuralNetwork():
    def count_params(self, layers):
        nparams = 0
        for l in range(1, len(layers)):
            nparams = nparams + (layers[l-1] * layers[l])
            nparams = nparams + layers[l]
        return nparams
    
    def __init__(self,
                 layers  = None,
                 hidden_activation = relu, 
                 output_activation = softmax,
                 loss = cross_entropy_loss,
                 random_seed = 42):
        np.random.seed(random_seed)
        nparams = self.count_params(layers)
        
        self.nlayers = len(layers)
        self.layers = layers
        self.hact = relu
        self.oact = softmax
        self.loss = cross_entropy_loss
        
        self.W = [None for _ in range(self.nlayers)]
        self.b = [None for _ in range(self.nlayers)]
        self.gW = [None for _ in range(self.nlayers)]
        self.gb = [None for _ in range(self.nlayers)]
        
        self.theta = np.random.normal(0, 1, size = nparams)
        self.gtheta = np.random.normal(0, 1, size = nparams)
        start, end = 0, 0
        for l in range(1, self.nlayers):
            end = end + (self.layers[l-1] * self.layers[l])
            self.W[l] = self.theta[start:end].reshape(self.layers[l-1], self.layers[l])
            self.gW[l] = self.gtheta[start:end].reshape(self.layers[l-1], self.layers[l])
            start, end = end, end + self.layers[l]
            self.b[l] = self.theta[start:end]
            self.gb[l] = self.gtheta[start:end]
            start = end

    def forward(self, X):
        self.A = [None for _ in range(self.nlayers)]
        self.H = [None for _ in range(self.nlayers)]
        self.H[0] = X
        self.A[0] = X
        for l in range(1, self.nlayers):
            self.A[l] = (self.H[l-1] @ self.W[l]) + self.b[l]
            self.H[l] = self.hact(self.A[l])
        self.H[-1] = softmax(self.A[-1])
        return self.H[-1]
    
    def backward(self, X, Y, Y_hat):
        gA = [None for _ in range(self.nlayers)]
        gH = [None for _ in range(self.nlayers)]
        gA[-1] = Y_hat - Y
        for l in range(self.nlayers - 1, 0, -1):
            self.gW[l][:,:] = self.H[l-1].T @ gA[l]
            self.gb[l][:] = np.sum(gA[l].T, axis = 1)
            gH[l-1] = gA[l] @ self.W[l].T
            gA[l-1] = grad_relu(self.A[l-1]) * gH[l-1]
    
    def fit(self, 
            X, 
            Y, 
            lr = 0.01, 
            batch_size = 10, 
            epochs = 5):
        self.losses = []
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            loss = self.loss(Y, Y_hat)
            self.losses.append(loss)
            rand_indexes = np.random.permutation(X.shape[0])
            X_shuffled = X[rand_indexes]
            Y_shuffled = Y[rand_indexes]
            for i in range(0, X.shape[0], batch_size):
                Y_hat = self.forward(X_shuffled[i:i+1])
                self.backward(X, Y_shuffled[i:i+1], Y_hat)
                self.theta[:] =  self.theta - lr*self.gtheta
                
    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis = 1)