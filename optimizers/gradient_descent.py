import numpy as np

class GradientDescent():
    def __init__(self, loss = 'SSE',
                penalty = None, alpha = 0):
        self.loss = loss
        self.penalty = penalty
        if self.penalty:
            self.alpha = alpha
        else:
            self.alpha = 0
    
    def predict(self, X, w):
        return X @ w
    
    def sse_gradient(self, X, w, y):
        return (X.T @ (self.predict(X, w) - y)) + self.alpha*w
    
    def sse_loss(self, X, w, y):
        assert X.shape[-1] == w.shape[0], 'Incompatible shapes'
        y_hat = self.predict(X, w)
        loss_ = np.sum(np.square(y_hat - y) + (self.alpha / 2)*np.dot(w.T,w))
        return loss_
    
    def descent(self, X, y,
               verbose, epochs, lr):
        w0 = np.random.normal(0, 1, size=(X.shape[1],1))
        self.all_weights = []
        if self.loss == 'SSE':
            for epoch in range(epochs):
              if verbose:
                print('The Current Loss is :', self.sse_loss(X, w0, y))
              self.all_weights.append(w0)
              w0 = w0 - lr*(self.sse_gradient(X, w0, y))
            print('The loss is', self.sse_loss(X, w0, y))
            return w0