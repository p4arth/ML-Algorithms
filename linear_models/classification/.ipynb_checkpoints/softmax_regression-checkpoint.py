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

from optimizers.gradient_descent import GradientDescent
from optimizers.mini_batch_gd import MiniBatchGD
from optimizers.stochastic_gd import StochasticGD
from preprocessing.add_dummy import add_dummy_feature

class SoftMaxRegression():
    def __init__(self,
                 penalty = None, 
                 alpha = 0,
                 threshold = 0.5, 
                 optimizer = 'GD'):
        self.penalty = penalty
        self.optimizer = optimizer
        self.threshold = threshold
        if self.penalty:
            self.alpha = alpha
        else:
            self.alpha = 0
    
    def softmax(self, z):
        softmax_vector = []
        for v in z:
            lc = np.exp(v)/np.sum(np.exp(v))
            softmax_vector.append(lc)
        return np.array(softmax_vector)
    
    def cross_entropy_loss(self, X, W, Y):
        assert X.shape[-1] == W.shape[0], 'Incompatible shapes'
        A = self.softmax(X @ W)
        A_log = np.log(A)
        log_proba = np.sum(Y * A_log, axis=1)
        return (1/Y.shape[0]) * np.sum(log_proba)
    
    def cross_entropy_gradient(self, X, W, Y):
        return X.T @ (self.softmax(X @ W) - Y)
    
    def fit(self,
            X,
            y,
            verbose = False, 
            epochs = 10,
            lr = 0.01, 
            batch_size = 0,
            fit_intercept = False):
        self.intercept = fit_intercept
        if self.intercept:
            X = add_dummy_feature(X)
        if self.optimizer == 'GD':
            gd = GradientDescent(penalty = self.penalty, 
                                 alpha = self.alpha, 
                                 activation_function = self.softmax,
                                 loss_function = self.cross_entropy_loss,
                                 gradient_function = self.cross_entropy_gradient)
            self.optimized_weights = gd.descent(X,
                                                y,
                                                verbose = verbose,
                                                epochs = epochs,
                                                lr = lr)
            self.weights = gd.all_weights
        elif self.optimizer == 'MBGD':
            mbgd = MiniBatchGD(penalty = self.penalty, 
                               alpha = self.alpha, 
                               activation_function = self.softmax,
                               loss_function = self.cross_entropy_loss,
                               gradient_function = self.cross_entropy_gradient)
            self.optimized_weights = mbgd.descent(X,
                                                  y, 
                                                  verbose,
                                                  epochs=epochs, 
                                                  batch_size = batch_size)
            self.weights = mbgd.all_weights
        elif self.optimizer == 'SGD':
            sgd = StochasticGD(penalty = self.penalty, 
                               alpha = self.alpha, 
                               activation_function = self.softmax,
                               loss_function = self.cross_entropy_loss,
                               gradient_function = self.cross_entropy_gradient)
            self.optimized_weights = sgd.descent(X,
                                                 y, 
                                                 verbose, 
                                                 epochs=epochs)
            self.weights = sgd.all_weights
    
    def predict_probas(self, X):
        if self.intercept:
            X = add_dummy_feature(X)
        A_p = self.softmax(X@self.optimized_weights)
        probas = np.max(A_p, axis = 1)
        return probas
    
    def predict(self, X):
        if self.intercept:
            X = add_dummy_feature(X)
        A_p = self.softmax(X@self.optimized_weights)
        classes = np.argmax(A_p, axis = 1)
        return classes
