
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import math
from __future__ import division, print_function




#Splits data set into training and test subsets
def splitting_data(X, y, test_size = 0.5, shuffle = True, seed = None):
    
    if shuffle:
        X, y = shuffle(X, y , seed)
        
    #split training data from test data by test_size factor.
    split = len(y) - int( len(y) // (1 / test_size) )
   
    X_train_set = X[ :split] #(elements before split factor)
    X_test_set = X[split: ] #(elements after split factor)
    y_test_set = y[split: ]
    y_train_set = y[ :split]
    
    return X_train_set, X_test_set, y_train_set, y_test_set
    
    #can also use train_test_split() method of sklearn directly
    
    
    
    
#computes accuracy of y_trueoriginal y of training data w.r.t predicted y
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis = 0)/ len(y_true)
    return accuracy
    #can use : " accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)" method of sklearn directly
    
  
    
#Decision Stump: Decision tree with one single split used as a weak classifier for adaboost algorithm   
class DecisionStump():
    def _init_(self):
        
        #Threshold split: Determines is the given sample can be classified as -1 or 1 given a threshold
        self.polarity = 1
        
        #Index of the feature that will be used to classify the data
        self.feature_index = None
        
        #Threshold value is the value the split will be made on by comparing it with data
        self.threshold = None
        
        #Coeffecient value calculated after every weak classification to measure accuracy of the classifier
        #alpha = 1/2 * ln( (1- error) / error )
        self.alpha = None
   
    
    
    
"""Adaboost is a boosting algorithm that uses number of weak classifiers in ensemble to produce a strong classifier. 
Here Adaboost uses Decision stumps as weak classifiers."""
class Adaboost():
    """Required Parameters:
        weak_classifiers (int):
            Gives number of weak classifiers used before producing the strong classifier. """
            

    def _init_(self, weak_classifiers = 5):
        self.weak_classifiers  = weak_classifiers 
        
    def fit(self, X, y):
        samples, features = np.shape(X)
        
        # initializing weights w = 1/N where N = total samples
        w = np.full(samples, (1/samples))
        #np.full(shape, fill_value): Return a new array of given shape and type, filled with fill_value. 
        
        #initializing empty list for weak classifiers
        self.classifiers = []
        
        #Iterate X through the weak classifiers
        for i in range(self.weak_classifiers):
            classifier = DecisionStump()
            
            #minimum error for using a certain feature value as threshold for predicting sample label 
            minimum_error = float('inf')
            #float('inf'): It acts as an unbounded upper value for comparison
            
            #Iterate through all feature values and see which value of a particular 
            #feature makes the best threshold for predicting y
            for feature_i in range(features):
                
                #extracting the values of a particular feature from data set X
                feature_vals = np.expand_dims(X[:, feature_i], axis = 0)
                """np.expand_dims(a, axis): Expand the shape of an array. Insert a new axis, corresponding to a given position in the array shape.
                >>> y = np.expand_dims(x, axis=0)
                >>> y
                array([[1, 2]])
                >>> y.shape
                (1, 2)  """
                
                #Removing duplicate feature values of a feature
                unique_vals = np.unique(feature_vals)
                
                #Test every feature value as threshold and choose the best suited
                
                for threshold in unique_vals:
                    p = 1
                    #set all predictions to 1 intially ie, set elements of prediction array to 1 (can originally be 1 or -1 though)
                    prediction = np.ones(np.shape(y))
                    
                    #Label the samples with values less than threshold to be -1 
                    prediction[X[:, feature_i] < threshold] =-1
                    
                    #Error = sum of weights of misclassified samples of data set X
                    error = sum(w[y != prediction])
                    
                    #if the error is greater than 50%, flip the polarity ie if classified as 1, flip it to -1 and vice versa
                    #eg if error = 0.8 => (1 - error) = 0.2
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                        
                        
                    #if error is small , there is no need to change configuration or no need to flip the polarity
                    if error < minimum_error:
                        classifier.polarity = p
                        classifier.threshold = threshold
                        classifier.feature_index = feature_i
                        minimum_error = error
                        
                
                #alpha = Coeffecient value calculated after every weak classification to measure accuracy of the classifier
                #alpha = 1/2 * ln( (1- error) / error )
                #using 1e-10 to prevent the denominator from being zero
                classifier.alpha = 0.5 * math.log(1.0 - minimum_error) / (minimum_error + 1e-10)
                
                #set all predictions to 1 intially
                predictions = np.ones(np.shape(y))
                
                #negative_index: indices where sample values are less than threshold values
                negative_index = ( classifier.polarity * X [: classifier.feature_index] < classifier.polarity * classifier.threshold )
                predictions[negative_index] = -1
                
                
                #Add predictions weighted by the classifiers alpha (alpha indicates classifier's accuracy)
                #final y prediction or hypothesis = sign * sum (alpha * predictions)
                y_prediction += classifier.alpha * predictions
                
                return y_prediction
     
                    
                
                
                
                
                
                
                
                
            
            
            
            
            
            
            
            
    
        
        
        
        
    