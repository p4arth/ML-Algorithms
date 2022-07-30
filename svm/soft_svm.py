import numpy as np
class SoftMarginSVM():
    def __init__(self, C = 1.0, random_seed = 42):
        np.random.seed(random_seed)
        self.C = C
    
    def loss(self, X, y):
        margin = (self.w.T @ self.w) / 2
        hinge_loss = np.sum(np.maximum(0, 1 - y*(X @ self.w + self.b)))
        return margin + self.C * hinge_loss
    
    def grad_loss(self, X, y):
        lin_comb = 1 - y*(X @ self.w + self.b)
        indic = np.where(lin_comb > 0, 1, 0)
        dJdw = self.w - self.C * np.sum((indic*y).reshape(-1,1) * X)
        dJdb = -self.C * np.sum(indic * y)
        return dJdw, dJdb
    
    def fit(self, X, y, 
            lr = 0.01, 
            epochs = 100,
            verbose = False):
        self.w = np.random.normal(0, 1, size = (X.shape[1]))
        self.b = np.random.normal(0, 1, size = 1)
        self.losses = []
        for epoch in range(epochs):
            epoch_loss = self.loss(X, y)
            if verbose:
                if epoch % 50 == 0:
                    print(f'The epoch {epoch} loss is {epoch_loss}')
            self.losses.append(epoch_loss)
            dJdw, dJdb = self.grad_loss(X, y)
            self.w = self.w - lr * dJdw
            self.b = self.b - lr * dJdb
            
    def predict(self, X):
        return np.sign(X @ self.w + self.b)