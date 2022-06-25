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

from preprocessing.add_dummy import add_dummy_feature
from linear_models.regression.linear_regression import LinearRegression
from optimizers.gradient_descent import GradientDescent
from optimizers.mini_batch_gd import MiniBatchGD
from optimizers.stochastic_gd import StochasticGD

class LeastSquareClassifier():
    def __init__(self, 
                 n_classes = 2, 
                 optimizer = 'GD', 
                 random_seed = 42):
        self.n_classes = n_classes
        self.optimizer = optimizer
        
    def predict_training(self, X, w):
        return X @ w
        
    def loss(self,X, w, y):
        loss_ = (1/2) * np.sum(np.square(self.predict_training(X, w) - y)) 
        return loss_
    
    def gradient(self, X, w, y):
        return  X.T @ (self.predict_training(X, w)  - y)
    
    def fit(self, 
              X, 
              y, 
              epochs = 200, 
              batch_size = 0, 
              learning_rate = 0.001, 
              verbose = False,
              penalty = 0):
        # TO-DO: Add intercept param and other optimizers
        if self.optimizer == 'GD':
            gd = GradientDescent(penalty = penalty, 
                                 alpha = 0, 
                                 loss_function = self.loss, 
                                 gradient_function = self.gradient)
            self.optimized_weights = gd.descent(X,
                                                y,
                                                verbose = verbose,
                                                epochs = epochs,
                                                lr = learning_rate)
            self.weights = gd.all_weights
    
    def predict(self, X):
        
        predictions = (X @ self.optimized_weights)
        return np.argmax(predictions, axis=1)
                    
            
                
            
            
            

