import numpy as np

class BaseOptimizers():
  def __init__(self, optimizer = 'GD'):
    self.optimizer = optimizer
  
  def loss(self, X, w, y):
    assert X.shape[-1] == w.shape[0], 'Incompatible shapes'
    y_hat = X @ w
    loss_ = np.sum(np.square(y_hat - y))
    return loss_

  def lr_schedule(self,t):
    t0, t1 = 200, 100000
    return t0 / (t + t1)
  
  def gradient(self, X, w, y):
    assert X.shape[-1] == w.shape[0], 'Incompatible shapes'
    return X.T @ ((X @ w) - y)
  
  def gradient_descent(self, X, y, 
                       verbose, epochs, lr):
    w0 = np.random.normal(0, 1, size=(X.shape[1],1))
    self.all_weights = []
    for epoch in range(epochs):
      if verbose:
        print('The Current Loss is :', self.loss(X, w0, y))
      self.all_weights.append(w0)
      w0 = w0 - lr*(self.gradient(X, w0, y))
    return w0

  def mini_batch_gd(self, X, y,
                    verbose, epochs, batch_size):
    w0 = np.random.normal(0, 1, size=(X.shape[1],1))
    t = 0
    self.all_weights = []
    for epoch in range(epochs):
      random_indices = np.random.permutation(X.shape[0])
      X_shuffled = X[random_indices]
      y_shuffled = y[random_indices]
      for i in range(0, X.shape[0], batch_size):
        t = t + 1
        X_temp = X_shuffled[i:i+batch_size]
        y_temp = y_shuffled[i:i+batch_size]
        lr = self.lr_schedule(t)
        self.all_weights.append(w0)
        w0 = w0 - lr*(self.gradient(X_temp, w0, y_temp))
      if verbose:
          print(f'Epoch {epoch} Loss is :', self.loss(X, w0, y))
    return w0

  def stochastic_gd(self, X,  
                    y, verbose, epochs):
    w0 = np.random.normal(0, 1, size=(X.shape[1],1))
    t = 0
    self.all_weights = []
    for epoch in range(epochs):
      for i in range(X.shape[0]):
        random_index = np.random.randint(X.shape[0])
        t = t + 1
        X_temp = X[random_index:random_index+1]
        y_temp = y[random_index:random_index+1]
        lr = self.lr_schedule(t)
        self.all_weights.append(w0)
        w0 = w0 - lr*(self.gradient(X_temp, w0, y_temp))
      if verbose:
          print(f'Epoch {epoch} Loss is :', self.loss(X, w0, y))
    return w0