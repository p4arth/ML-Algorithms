import numpy as np

class MultinomialNB():
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        