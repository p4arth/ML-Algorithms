import os
import sys
sys.path.insert(0, os.path.abspath(".."))
import numpy as np

class GradientBoostingRegressor():
    def __init__(self, 
                 n_estimators = 10,
                 max_depth = 2,
                 lr = 0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
    
    def fit_predict(self, 
                    Xtrain, ytrain,
                    Xtest, ytest):
        self.ytrain_mean = np.repeat(np.mean(ytrain), ytrain.shape[0])
        yhat_test_mean = np.repeat(np.mean(ytrain), ytest.shape[0])
        residuals = ytrain - self.ytrain_mean
        for i in range(self.n_estimators):
            estimator = DecisionTreeRegressor(max_depth = self.max_depth)
            estimator.fit(Xtrain, residuals)
            self.ytrain_mean = self.ytrain_mean + self.lr * estimator.predict(Xtrain)
            yhat_test_mean = yhat_test_mean + self.lr * estimator.predict(Xtest)
            residuals = ytrain - self.ytrain_mean
        return yhat_test_mean