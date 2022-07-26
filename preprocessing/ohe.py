import numpy as np
def OneHotEncoder(labels):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels) 
    return np.eye(n_labels)[labels]