from gradient_optimizers import BaseOptimizers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class LinearRegression(BaseOptimizers):
  def __init__(self,
               optimizer = 'GD',
               random_seed = 42):
    super().__init__(optimizer)
    self.random_seed = random_seed
  
  def add_dummy_feature(self, X):
    matrix_dummy = np.hstack((np.ones((X.shape[0], 1),
                                            dtype = X.dtype), 
                                            X))
    return matrix_dummy
  
  def preprocess(self, X):
    X = self.add_dummy_feature(X)
    return X
  
  def train(self,
            X_train,
            y_train,
            epochs = 200,
            batch_size = 100,
            learning_rate = 0.001,
            verbose = False):
    assert batch_size < X_train.shape[0], 'batch size must be smaller than the number of data points'
    X_train = self.preprocess(X_train)
    if self.optimizer == 'GD':
      self.optimized_weights = self.gradient_descent(X_train, y_train,
                                                    verbose, epochs,
                                                    learning_rate)                          
    elif self.optimizer == 'MBGD':
      self.optimized_weights = self.mini_batch_gd(X_train, y_train,
                                                  verbose, epochs=epochs,
                                                  batch_size = batch_size)                                     
    else:
      self.optimized_weights = self.stochastic_gd(X_train, y_train,
                                                  verbose, epochs=epochs)
    self.weights = self.all_weights
                                                  
  def predict(self, X):
    X = self.add_dummy_feature(X)
    assert X.shape[-1] == self.optimized_weights.shape[0], 'Incompatible Shapes'
    return X @ self.optimized_weights


# X, y = make_regression(n_samples = 10000)
# y = y.reshape(-1,1)
# x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

# model_gd = LinearRegression(optimizer = 'SGD')
# model_gd.train(x_train, y_train, epochs = 10, verbose = True, learning_rate = 0.0001)
# error = model_gd.predict(x_test) - y_test
# sum_squared_error = np.sum(np.square(error))
# print(sum_squared_error)