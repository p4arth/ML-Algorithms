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

class LogisticRegression():
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
            
    def sigmoid(self, vector):
        return 1/(1 + np.exp(-1*vector))
    
    def neg_log_loss(self, X, w, y):
        assert X.shape[-1] == w.shape[0], 'Incompatible shapes'
        y_hat = self.sigmoid(X@w)
        loss_ = (1/X.shape[0]) * (-1) * np.sum(y*(np.log(y_hat)) + (1-y)*np.log(1 - y_hat) + (self.alpha/2) * np.dot(w.T, w))
        return loss_
    
    def neg_log_gradient(self, X, w, y):
        return (X.T @ (self.sigmoid(X@w) - y)) + self.alpha*w
    
    def fit(self, 
            X, 
            y,
            verbose, 
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
                                 loss_function = self.neg_log_loss, 
                                 gradient_function = self.neg_log_gradient)
            self.optimized_weights = gd.descent(X,
                                                y, 
                                                verbose = verbose, 
                                                epochs = epochs, 
                                                lr = lr)
            self.weights = gd.all_weights
        elif self.optimizer == 'MBGD':
            mbgd = MiniBatchGD(penalty = self.penalty, 
                               alpha = self.alpha, 
                               loss_function = self.neg_log_loss, 
                               gradient_function = self.neg_log_gradient)
            self.optimized_weights = mbgd.descent(X,
                                                  y, 
                                                  verbose, 
                                                  epochs=epochs, 
                                                  batch_size = batch_size)
            self.weights = mbgd.all_weights
        elif self.optimizer == 'SGD':
            sgd = StochasticGD(penalty = self.penalty, 
                               alpha = self.alpha, 
                               loss_function = self.neg_log_loss, 
                               gradient_function = self.neg_log_gradient)
            self.optimized_weights = sgd.descent(X, 
                                                 y, 
                                                 verbose, 
                                                 epochs=epochs)
            self.weights = sgd.all_weights

    def predict(self, X):
        if self.intercept:
            X = add_dummy_feature(X)
        return np.where(self.sigmoid(X@self.optimized_weights) > self.threshold, 1, 0)