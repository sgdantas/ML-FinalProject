#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:47:12 2018

@author: sgdantas
"""

import numpy as np
from sklearn import svm
from load_show_data import labels, features
from sklearn.model_selection import train_test_split

X,X_test,y,y_test = train_test_split(features,labels,test_size = 0.2, random_state = 20)

def svm_kernels():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    for a in kernels:
        clf = svm.SVC(kernel = a,degree = 10,gamma='scale')
        clf.fit(X, np.argmax(y,axis = 1))  
        
        print(clf.score(X_test,np.argmax(y_test,axis = 1)))
    

def svm_poly():
    
    deg = [3,5,7,10,15,20]
    
    for a in deg:
        X,X_test,y,y_test = train_test_split(features,labels,test_size = 0.2, random_state = 20)
        clf = svm.SVC(kernel = 'poly',degree = a,gamma='scale')
        clf.fit(X, np.argmax(y,axis = 1))  
        
        print(clf.score(X_test,np.argmax(y_test,axis = 1)))
    




