import sys
import os
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass

from linear_models.regression.linear_regression import LinearRegression
import numpy as np
import pandas as pd

class LeastSquareClassifier():
    def __init__(self, 
                 n_classes = 2, 
                 optimizer = 'GD', 
                 random_seed = 42):
        self.n_classes = n_classes
        self.optimizer = optimizer
        
    def encode_labels(self, labels):
        n_labels = np.max(labels) + 1
        return np.eye(n_labels)[labels]
        
    def predict_training(self, X, w):
        predictions = (X @ w)
        return predictions
        
    def loss(self,X, w, y):
        loss_ = np.sum(np.square(self.predict_training(X, w) - y)) 
        return loss_
    
    def gradient(self, X, w, y):
        return X.T @ (self.predict_training(X, w)  - y) * (1/X.shape[0])
    
    def gradient_descent(self,
                         X, 
                         y, 
                         verbose, 
                         epochs,
                         lr):
        w0 = np.random.normal(0, 1, size=(X.shape[1],1))
        self.all_weights = []
        for epoch in range(epochs):
          if verbose:
            print('The Current Loss is :', self.loss(X, w0, y))
          self.all_weights.append(w0)
          w0 = w0 - lr*(self.gradient(X, w0, y))
        return w0
    
    def train(self, 
              X_train, 
              y_train, 
              epochs = 200, 
              batch_size = 100, 
              learning_rate = 0.001, 
              verbose = False):
        y_train = self.encode_labels(y_train)
        self.linreg = LinearRegression()
        X_train = self.linreg.add_dummy_feature(X_train)
        self.optimized_weights = self.gradient_descent(X_train, y_train,
                                                    verbose, epochs,
                                                    learning_rate)
    
    def predict(self, X):
        X = self.linreg.add_dummy_feature(X)
        predictions = (X @ self.optimized_weights)
        return np.array([0 if val >= self.optimized_weights[0]
                         else 1 for val in predictions]).reshape(-1,1)
                    
            
                
            
            
            

