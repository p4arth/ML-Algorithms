import numpy as np
def add_dummy_feature(X):
    '''
    Adds a dummy variable to the feature matrix.
    '''
    matrix_dummy = np.hstack((np.ones((X.shape[0], 1),
                                            dtype = X.dtype), 
                                            X))
    return matrix_dummy