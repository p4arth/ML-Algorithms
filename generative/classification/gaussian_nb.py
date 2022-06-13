import numpy as np

class GaussianNB():
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        
        self.n_classes = len(self.classes)
        self._mean = np.zeros((self.n_samples, self.n_features), dtype = np.float64)
        self._var = np.zeros((self.n_samples, self.n_features), dtype = np.float64)
        self._priors = np.zeros(self.n_classes, dtype = np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self._mean[idx,:] = X_c.mean(axis = 0)
            self._var[idx,:] = X_c.var(axis = 0)
            self._priors[idx] = X_c.shape[0] / float(self.n_samples)
    
    def normal_pdf(self, class_id, X):
        mean_vector = self._mean[class_id]
        covar_mat = np.diag(self._var[class_id])
        normal_denom = np.power(2*np.pi, X.shape[0]/2) * np.power(np.linalg.det(covar_mat), 1/2)
        X_minus_mu = X - mean_vector
        return (1/normal_denom) * np.exp(-(1/2)*(X_minus_mu).T @ (np.linalg.inv(covar_mat)) @ (X_minus_mu))
    
    def posteriors(self, X):
        self.log_post = np.zeros((X.shape[0], self.n_classes), dtype = np.float64)
        for x_id, x in enumerate(X):
            for idx, c in enumerate(self.classes):
                self.log_post[x_id, c] = np.log(self.normal_pdf(idx, x)) + np.log(self._priors[idx])
    
    def predict(self, X):
        self.posteriors(X)
        return np.argmax(self.log_post, axis = 1)