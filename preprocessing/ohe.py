import numpy as np
def OneHotEncoder(labels):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels) 
    if 0 in unique_labels:
        return np.eye(n_labels)[labels]
    else:
        return np.eye(n_labels+1)[labels][:,1:]