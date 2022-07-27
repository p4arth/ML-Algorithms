import sys
import os
from pathlib import Path
import numpy as np
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:
    pass

from scipy import stats
from trees.decision_tree_classifier import DecisionTreeClassifier
import concurrent.futures
class RandomForestClassifier():
    def __init__(self, 
                 n_estimators = 3, 
                 min_samples_split = 4,
                 max_depth = 5,
                 max_features = None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.estimators = []
    
    def _fit(self, X, y):
        estimator = DecisionTreeClassifier(min_samples_split = self.min_samples_split, 
                                           max_depth = self.max_depth,
                                           max_features = self.max_features)
        estimator.fit(X, y)
        return estimator
    
    def fit(self, X, y):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            res = [executor.submit(self._fit, X, y) for _ in range(self.n_estimators)]
        self.estimators = []
        for f in concurrent.futures.as_completed(res):
            self.estimators.append(f.result())
            
    def predict(self, X):
        predictions = []
        for estimator in self.estimators:
            predictions.append(np.transpose(estimator.predict(X)))
        return stats.mode(predictions)[0][0]