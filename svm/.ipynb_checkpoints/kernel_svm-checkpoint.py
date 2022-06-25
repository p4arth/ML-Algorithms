import numpy as np
class KernelSVM():
    def __init__(self, 
                 kernel='polynomial', 
                 C = 0.01,
                 sigma = 0.1, 
                 degree = 2):
        self.C = C
        if kernel == 'polynomial':
            self.kernel = self.polynomial_kernel
            self.degree = degree
        else:
            self.kernel = self.rbf_kernel
            self.sigma = sigma
    
    def polynomial_kernel(self, X1, X2):
        return (X1.dot(X2.T))**self.degree
    
    def rbf_kernel(self, X1, X2):
        return np.exp(-(1/self.sigma**2) * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis,:],
                                                          axis = 2)**2)
    
    def fit(self, X, y, lr = 0.01, epochs = 100):
        