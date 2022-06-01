import numpy as np

class Perceptron():
    def __init__(self):
        pass
#     def misclassified(self, y_hat, y_true):
        
    def fit(self, X_train, y_train, epochs = 10):
        self.w0 = np.zeros(X_train.shape[1]).reshape(-1,1)
        all_weights = [self.w0]
        for epoch in range(epochs):
            misclassified = 0
            for i in range(X_train.shape[0]):
                if y_train[i] @ (X_train[i] @ self.w0) <= 0:
                    self.w0 = self.w0 + np.dot(X_train[i].reshape(-1,1),
                                               y_train[i].reshape(-1,1))                
                    all_weights.append(self.w0)
                    misclassified = misclassified + 1
            print(f'The # misclassified points for epoch {epoch} are {misclassified}')
            if not misclassified:
                break
    
    def predict(self, X):
        z = X @ self.w0
        return np.where(z>=0, 1, -1)