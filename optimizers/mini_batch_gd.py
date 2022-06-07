import numpy as np

class MiniBatchGD():
    def __init__(self, loss_function = None,
                 gradient_function = None,
                penalty = None, alpha = 0):
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
    
    def descent(self, X, y,
                      verbose, epochs,
                      batch_size):
        w0 = np.random.normal(0, 1, size=(X.shape[1],1))
        t = 0
        self.all_weights = []
        for epoch in range(epochs):
            random_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[random_indices]
            y_shuffled = y[random_indices]
            for i in range(0, X.shape[0], batch_size):
                t = t + 1
                lr = self.lr_schedule(t)
                X_temp = X_shuffled[i:i+batch_size]
                y_temp = y_shuffled[i:i+batch_size]
                self.all_weights.append(w0)
                w0 = w0 - lr*(self.gradient(X_temp, w0, y_temp))
            if verbose:
                print(f'Epoch {epoch} Loss is :', self.loss(X, w0, y))
        print('The loss is', self.loss(X, w0, y))
        return w0