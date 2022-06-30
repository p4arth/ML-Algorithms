'''
This implementation is inspired from https://youtu.be/sgQAhG5Q7iY

This implementaion only works on numerical features
so please encode the categorical features before providing
to the methods. Also please provide all the data as
a numpy array.

If you find any erros in the code please open an
issue in the github repository.
'''

from scipy import stats as st
import numpy as np

class Node():
    '''This function creates a single node of the decision tree'''
    def __init__(self, 
                 attribute_name = None,
                 threshold = None,
                 left_child = None, 
                 right_child = None):
        self.attribute_name = attribute_name
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child


class DecisionTreeClassifier():
    def __init__(self, 
                 min_samples_split = 4,
                 max_depth = 5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def make_tree(self,  X, y, depth=0):
        '''This function creates the decision tree recursively'''
        if len(X) > self.min_samples_split and self.max_depth != depth:
            best_split = self.get_best_split(X, y)
            if best_split['Information_gain'] > 0:
                left_child = self.make_tree(best_split['X_left_data'], 
                                            best_split['y_left_data'],
                                            depth + 1)
                right_child = self.make_tree(best_split['X_right_data'], 
                                             best_split['y_right_data'],
                                             depth + 1)
                return Node(
                    attribute_name = best_split['Attribute'],
                    threshold = best_split['Threshold'],
                    left_child = left_child,
                    right_child = right_child
                )

        leaf_value = self.get_leaf_value(y)
        return Node(attribute_name = leaf_value)
    
    def get_best_split(self, X, y):
        '''
        This function returns the best split of our data.
        The function will go through each attribute in the data
        and try to find the best split based on each data
        point in every attribute. It will find a split that has
        the maximum information gain.
        
        The function returns a dictionary containing the best split
        which is used in the make_tree method.
        '''
        max_information_gain = -float('inf')
        best_threshold = 0
        attribute = 0
        
        for i in range(X.shape[1]):
            uniq_thresholds = np.unique(X[:, i])
            for threshold in uniq_thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, i, threshold)
                information_gain = self.calculate_information_gain(y, y_left, y_right)
                if information_gain > max_information_gain:
                    attribute = i
                    max_information_gain = information_gain
                    best_threshold = threshold
                    X_left_data = X_left
                    X_right_data = X_right
                    y_left_data = y_left
                    y_right_data = y_right
        best_split = {
            'Attribute': attribute,
            'Information_gain': max_information_gain,
            'Threshold': best_threshold,
            'X_left_data': X_left_data,
            'X_right_data': X_right_data,
            'y_left_data': y_left_data,
            'y_right_data': y_right_data
        }
        return best_split
    
    def split(self, X, y, i, threshold):
        '''
        This function partitions the data into two halves
        based on a feature and a threhold value.
        The function returns the partitioned data.
        '''
        temp_d = np.column_stack((X, y))
        d_left = temp_d[temp_d[:, i] <= threshold]
        d_right = temp_d[temp_d[:, i] > threshold]
        return d_left[:,:-1], d_right[:,:-1], d_left[:,-1], d_right[:,-1]
    
    def get_leaf_value(self, y):
        '''
        This function returns what class label should the
        leave of the decision tree contain. It chooses the
        mode value in the paritioned label vector.
        
        '''
        try:
            return st.mode(y)[0][0]
        except:
            return None
    
    def calculate_information_gain(self, y, y_left, y_right):
        '''
        This function calculates and returns the informaion gain
        based on the gini index that we would get from a split.
        '''
        # TO-DO : Add entropy information gain
        wleft = len(y_left)/len(y)
        wright = len(y_right)/len(y)
        return self.gini_index(y) - ((wleft*self.gini_index(y_left)) + (wright*self.gini_index(y_right)))
    
    def gini_index(self, y):
        '''
        This function calculates the gini index for a label vector.
        '''
        classes = np.unique(y)
        gini = 0
        for cl in classes:
            pr = len(y[y==cl]) / len(y)
            gini = gini + pr**2
        return 1 - gini
    
    def exec_(self, tree=None, indent = ' '):
        if tree.left_child == None and tree.right_child == None:
            print(f'--> (Class: {tree.attribute_name})')
        else:
            print("if Attr "+str(tree.attribute_name), "<=", tree.threshold)
            print("%sleft: "%(indent), end='')
            self.exec_(tree.left_child, indent + indent)
            print("%sright: "%(indent), end='')
            self.exec_(tree.right_child, indent + indent)
    
    def print_tree(self):
        '''
        This function calls the exec_ function
        which prints the fitted decision tree.
        '''
        self.exec_(self.tree)
            
    def traverse_tree(self, node, z):
        '''
        This funcion is used to traverse the fitted decision tree when 
        a new datapoint has to be classified.
        '''
        if node.left_child == None and node.right_child == None:
            return node.attribute_name
        if z[node.attribute_name] <= node.threshold:
            return self.traverse_tree(node.left_child, z)
        else:
            return self.traverse_tree(node.right_child, z)
    
    def fit(self, X, y):
        '''
        This function fits the decision tree to the data.
        '''
        assert X.shape[0] == y.shape[0], 'X and y must have the same dimenions along the rows'
        self.tree = self.make_tree(X, y)
        
    def predict(self, X):
        '''
        This function predicts the class labels of all
        the data points present in the feature matrix X.
        '''
        res = []
        for i in range(X.shape[0]):
            node_value = self.traverse_tree(self.tree, X[i])
            res.append(node_value)
        return res