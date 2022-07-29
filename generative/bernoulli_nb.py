import numpy as np

class BernoulliNB():
    def __init__(self, alpha = 1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        self.w = np.zeros((n_classes, n_features), dtype = np.float64)
        self.c_priors = np.zeros(n_classes, dtype = np.float64)
        
        for c in range(classes):
            X_c = X[y == c]
            self.w[c, :] = (np.sum(X_c, axis = 0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha)
            self.c_priors[c] = (X_c.shape[0] + self.alpha)/(float(n_samples) + n_classes*self.alpha)
    
    def log_likelihood(self, X):
        return X@(np.log(self.w).T) + (1-X)@np.log((1-self.w).T) + np.log(self.c_priors)
    
    def predict_proba(self, X):
        log_probs = self.log_likelihood(X)
        return np.exp(log_probs) / np.expand_dims(np.sum(np.exp(log_probs), axis = 1), axis = 1)
    
    def predict(self, X):
        return np.argmax(self.log_likelihood(X), axis = 1)