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

from optimizers.gradient_descent import GradientDescent
from optimizers.mini_batch_gd import MiniBatchGD
from optimizers.stochastic_gd import StochasticGD
from preprocessing.add_dummy import add_dummy_feature

class SoftMaxRegression():
    def __init__(self,
                 penalty = None, 
                 alpha = 0,
                 threshold = 0.5, 
                 optimizer = 'GD'):
        '''
        This function initializes the softmax regression
        object.
        Parameters:
            penalty: 
                The type of regularization that can 
                be added to the model. Valid Values
                "l2" and "None". Default: None.
            alpha:
                The regularization rate. Default: 0.
            threshold:
                The threshold value above which a data
                point is classified belonging to a
                particular class. Default: 0.5.
            optimizer:
                The type of optimizer to be used. Valid
                values are "GD", "MBGD", and "SGD".
                Default: "GD".
        '''
        self.penalty = penalty
        self.optimizer = optimizer
        self.threshold = threshold
        if self.penalty:
            self.alpha = alpha
        else:
            self.alpha = 0
    
    def softmax(self, z):
        '''
        This applies the softmax normalization to
        a vector.
        Parameters:
            z:
                The matrix that needs to be soft-maxxed.
        '''
        softmax_vector = []
        # Softmax normalization applied on every vector in z.
        for v in z:
            lc = np.exp(v)/np.sum(np.exp(v))
            softmax_vector.append(lc)
        return np.array(softmax_vector)
    
    def cross_entropy_loss(self, X, W, Y):
        '''
        This function calculates the cross-entropy loss.
        '''
        assert X.shape[-1] == W.shape[0], 'Incompatible shapes'
        # Softmax after multipying X with W.
        A = self.softmax(X @ W)
        A_log = np.log(A)
        log_proba = np.sum(Y * A_log, axis=1)
        return (1/Y.shape[0]) * np.sum(log_proba)
    
    def cross_entropy_gradient(self, X, W, Y):
        '''
        This function calculates the gradient
        of the softmax function.
        '''
        return X.T @ (self.softmax(X @ W) - Y)
    
    def fit(self,
            X,
            y,
            verbose = False, 
            epochs = 10,
            lr = 0.01, 
            batch_size = 0,
            fit_intercept = False):
        '''
        This function fits the softmax regression
        object.
        Parameters:
            X: 
                The feature matrix.
            y:
                The one hot encoded corresponding 
                class labels.
            verbose:
                The level of verbosity.
            epochs:
                The number of epochs for which the
                optimizer should run.
            lr:
                The learning rate for the optimizer.
            batch_size:
                The batch size to take while using
                the Mini Batch Gradient Descent.
            fit_intercept:
                If True it adds a dummy feature to
                the feature matrix X.
        '''
        self.intercept = fit_intercept
        if self.intercept:
            X = add_dummy_feature(X)
        if self.optimizer == 'GD':
            gd = GradientDescent(penalty = self.penalty, 
                                 alpha = self.alpha, 
                                 activation_function = self.softmax,
                                 loss_function = self.cross_entropy_loss,
                                 gradient_function = self.cross_entropy_gradient)
            self.optimized_weights = gd.descent(X,
                                                y,
                                                verbose = verbose,
                                                epochs = epochs,
                                                lr = lr)
            self.weights = gd.all_weights
        elif self.optimizer == 'MBGD':
            mbgd = MiniBatchGD(penalty = self.penalty, 
                               alpha = self.alpha, 
                               activation_function = self.softmax,
                               loss_function = self.cross_entropy_loss,
                               gradient_function = self.cross_entropy_gradient)
            self.optimized_weights = mbgd.descent(X,
                                                  y, 
                                                  verbose,
                                                  epochs=epochs, 
                                                  batch_size = batch_size)
            self.weights = mbgd.all_weights
        elif self.optimizer == 'SGD':
            sgd = StochasticGD(penalty = self.penalty, 
                               alpha = self.alpha, 
                               activation_function = self.softmax,
                               loss_function = self.cross_entropy_loss,
                               gradient_function = self.cross_entropy_gradient)
            self.optimized_weights = sgd.descent(X,
                                                 y, 
                                                 verbose, 
                                                 epochs=epochs)
            self.weights = sgd.all_weights
    
    def predict_probas(self, X):
        '''
        This function returns the probability of each
        class label belonging to the corresponding class.
        '''
        if self.intercept:
            X = add_dummy_feature(X)
        A_p = self.softmax(X@self.optimized_weights)
        probas = np.max(A_p, axis = 1)
        return probas
    
    def predict(self, X):
        '''
        This function predicts the class to which a
        specific data point belongs to using the 
        optimized weights obtained from fit method.
        '''
        if self.intercept:
            X = add_dummy_feature(X)
        A_p = self.softmax(X@self.optimized_weights)
        classes = np.argmax(A_p, axis = 1)
        return classes
