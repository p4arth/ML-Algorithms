import numpy as np

class GradientDescent():
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
    
    def descent(self,
                X,
                y,
                verbose,
                epochs,
                lr):
        w0 = np.random.normal(0, 1, size=(X.shape[1], y.shape[1]))
        self.all_weights = []
        for epoch in range(epochs):
          if verbose:
            print('The Current Loss is :', self.loss(X, w0, y))
          self.all_weights.append(w0)
          w0 = w0 - lr*(self.gradient(X, w0, y))
        print('The loss is', self.loss(X, w0, y))
        return w0