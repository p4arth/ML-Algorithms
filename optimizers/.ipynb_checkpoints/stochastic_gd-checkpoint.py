import numpy as np

class StochasticGD():
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
    
    def lr_schedule(self,t):
        t0, t1 = 200, 100000
        return t0 / (t + t1)
    
    def descent(self, X, y, 
                verbose, epochs):
        w0 = np.random.normal(0, 1, size=(X.shape[1],1))
        t = 0
        self.all_weights = []
        if self.loss == 'SSE':
            for epoch in range(epochs):
              for i in range(X.shape[0]):
                random_index = np.random.randint(X.shape[0])
                t = t + 1
                X_temp = X[random_index:random_index+1]
                y_temp = y[random_index:random_index+1]
                lr = self.lr_schedule(t)
                self.all_weights.append(w0)
                w0 = w0 - lr*(self.sse_gradient(X_temp, w0, y_temp))
              if verbose:
                  print(f'Epoch {epoch} Loss is :', self.sse_loss(X, w0, y))
            print('The loss is', self.sse_loss(X, w0, y))
            return w0