import numpy as np

class StochasticGD():
    def __init__(self, 
                 penalty = None,
                 alpha = 0,
                 loss_function = None,
                 gradient_function = None,
                 activation_function = None,
                 ):
        self.loss = loss_function
        self.gradient = gradient_function
        self.penalty = penalty
        if self.penalty:
            self.alpha = alpha
        else:
            self.alpha = 0
            
    def lr_schedule(self,t):
        t0, t1 = 200, 100000
        return t0 / (t + t1)
    
    def descent(self, 
                X, 
                y, 
                verbose, 
                epochs):
        w0 = np.random.normal(0, 1, size=(X.shape[1],y.shape[1]))
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
        print('The loss is', self.loss(X, w0, y))
        return w0