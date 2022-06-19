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

class LinearRegression():
  def __init__(self,
               optimizer = 'GD',
               random_seed = 42):
    self.random_seed = random_seed
    self.optimizer = optimizer

  def preprocess(self, X):
    X = add_dummy_feature(X)
    return X

  def sse_gradient(self, X, w, y):
    return (X.T @ ((X @ w) - y)) + self.alpha*w
    
  def sse_loss(self, X, w, y):
    assert X.shape[-1] == w.shape[0], 'Incompatible shapes'
    y_hat = X @ w
    loss_ = np.sum(np.square(y_hat - y) + (self.alpha / 2)*np.dot(w.T,w))
    return loss_

  def fit(self,
            X_train,
            y_train,
            epochs = 200,
            batch_size = 0,
            learning_rate = 0.001,
            verbose = False,
            penalty = None,
            alpha = 0):
    assert batch_size < X_train.shape[0], 'batch size must be smaller than the number of data points'
    
    X_train = self.preprocess(X_train)
    self.alpha = alpha
    if self.optimizer == 'GD':
      gd = GradientDescent(penalty = penalty, 
                           alpha = self.alpha, 
                           loss_function = self.sse_loss, 
                           gradient_function = self.sse_gradient)
      self.optimized_weights = gd.descent(X_train, 
                                          y_train,
                                          verbose = verbose,
                                          epochs = epochs,
                                          lr = learning_rate)
      self.weights = gd.all_weights
    
    elif self.optimizer == 'MBGD':
      mbgd = MiniBatchGD(penalty = penalty, 
                         alpha = self.alpha, 
                         loss_function = self.sse_loss, 
                         gradient_function = self.sse_gradient) 
      self.optimized_weights = mbgd.descent(X_train, 
                                            y_train, 
                                            verbose, 
                                            epochs=epochs, 
                                            batch_size = batch_size)
      self.weights = mbgd.all_weights
    elif self.optimizer == 'SGD':
      sgd = StochasticGD(penalty = penalty, 
                         alpha = self.alpha, 
                         loss_function = self.sse_loss, 
                         gradient_function = self.sse_gradient) 
      self.optimized_weights = sgd.descent(X_train, 
                                           y_train, 
                                           verbose, 
                                           epochs=epochs)
      self.weights = sgd.all_weights
    
  def predict(self, X):
    X = self.add_dummy_feature(X)
    assert X.shape[-1] == self.optimized_weights.shape[0], 'Incompatible Shapes'
    predictions = X @ self.optimized_weights
    return predictions

