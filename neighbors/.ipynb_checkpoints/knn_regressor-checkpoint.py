import numpy as np
from scipy import stats as st

class KNeighborsRegressor():
    def __init__(self, 
                 n_neighbors = 3,
                 distance_metric = 'Euclidian'):
        assert distance_metric in ['Manhattan', 'Euclidian'], 'Wrong distance metric chosen. The valid distance metrics are ["Manhattan", "Euclidian"]'
        
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        
    def manhattan_distance(self, X_pred):
        self.distances = []
        for dp in self.X:
            difference_vector = X_pred - dp
            self.distances.append(np.sum(np.abs(difference_vector)))
        self.distances = np.array(self.distances)
        
    def euclidian_distance(self, X_pred):
        self.distances = []
        for dp in self.X:
            difference_vector = X_pred - dp
            self.distances.append(np.sum(np.square(difference_vector)))
        self.distances = np.array(self.distances)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X_pred):
        yhat = []
        for new_point in X_pred:
            if self.distance_metric == 'Manhattan':
                self.manhattan_distance(new_point)
            elif self.distance_metric == 'Euclidian':
                self.euclidian_distance(new_point)
            minimal_distance_idx = np.argpartition(self.distances,
                                                  self.n_neighbors)
            minimal_distance_idx = minimal_distance_idx[:self.n_neighbors]
            values = self.y[minimal_distance_idx]
            label = np.mean(values)
            yhat.append(label)
        return np.array(yhat)