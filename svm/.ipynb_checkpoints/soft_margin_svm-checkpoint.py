import numpy as np

class SoftMarginSVM():
    np.random.seed(42)
    def __init__(self, C = 0.01):
        self.C = C
    
    def accu(self, yt, yh):
        return np.sum(yt == np.sign(yh))
    
    def hinge_loss(self, X, w, b, y):
        loss = (1/2)*w.dot(w) + self.C * np.sum(
            np.maximum(0, 1 - (y * (X @ w + b)))
        )
        return loss
    
    def hinge_gradient(self, X, w, b, y):
        margin = y * (X @ w + b)
        misclassified_points = np.where(margin < 1)[0]
        dJdw = y[misclassified_points].dot(X[misclassified_points])
        dJdb = np.sum(y[misclassified_points])
        return (dJdw, dJdb)
    
    def fit(self, X, y, epochs = 100, lr = 0.01):
        w0 = np.random.normal(0, 1, size=(X.shape[1]))
        b = 0
        self.loss_array = []
        self.w_norms = []
        self.b_norm = []
        for _ in range(epochs):
            loss = self.hinge_loss(X, w0, b, y)
#             print('The current loss is:', loss)
            self.loss_array.append(loss)
            
            dJdw, dJdb = self.hinge_gradient(
                X, w0, b, y
            )
            
#             norm = np.linalg.norm()
            self.w_norms.append(np.linalg.norm(w0))
            self.b_norm.append(b)
                                
            
            wmid = w0 - (self.C * dJdw)
            bmid = self.C * dJdb
#             print(w0, b)
            w0 = w0 - lr*wmid
            b = b - lr*bmid
            