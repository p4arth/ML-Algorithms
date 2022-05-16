import numpy as np
from linear_regression import LinearRegression
import itertools
import functools

class PolynomialRegression():
    def __init__(self, degrees = 2,
                optimizer = 'GD',
                 random_seed = 42):
        self.degrees = degrees
        self.optimizer = optimizer
    
    def polynomial_transform(self, X):
        if X.shape == 1:
            X = X.reshape(-1,1)
        X = X.T
        transformed_features = []
        for degree in range(1, self.degrees+1):
            for item in itertools.combinations_with_replacement(X, degree):
                transformed_features.append(
                    np.array(functools.reduce(lambda x, y: x * y, item))
                )
        return np.array(transformed_features).T
    
    def train(self, X_train,
            y_train,
            epochs = 200,
            batch_size = 100,
            learning_rate = 0.001,
            verbose = False):
        X_train = self.polynomial_transform(X_train)
        self.model = LinearRegression(optimizer = self.optimizer)
        self.model.train(X_train, y_train,
            epochs = epochs,
            batch_size = batch_size,
            learning_rate = learning_rate,
            verbose = verbose)
        self.optimized_weights = self.model.optimized_weights
    
    def predict(self, X):
        X = self.polynomial_transform(X)
        self.model.predict(X)
        return self.model.predictions
        


        
        
        
        
