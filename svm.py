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



empty = np.empty((10,4))



def split_data():
    return train_test_split(features,labels,test_size = 0.2)


### SVM with different kernels

def svm_kernels(c,empty):
    for j in range(10):        
        X,X_test,y,y_test = split_data()
        kernels = ['linear', 'rbf', 'sigmoid', 'poly']
        
        for i in range(len(kernels)):
            clf = svm.SVC(C = c,kernel = kernels[i], gamma = 'scale')
            clf.fit(X, np.argmax(y,axis = 1))
            score = clf.score(X_test,np.argmax(y_test,axis = 1))
            #print(score)
            empty[j,i] = score
        
    return empty



## for different degrees for polynomial kernel

#def svm_poly():
#    
#    deg = [3,5,7,10,15,20]
#    
#    for a in deg:
#        X,X_test,y,y_test = split_data()
#        clf = svm.SVC(kernel = 'poly',degree = a)
#        clf.fit(X, np.argmax(y,axis = 1))  
#        
#        print(clf.score(X_test,np.argmax(y_test,axis = 1)))
#    




#### SVM for different C's

def svm_diff_C():
    cs = [0.1,0.25,0.5,1,2,5,10]
    
    for c in cs:
        print('using c = %.1f'%c)    
        results = svm_kernels(c,empty)
        print(np.mean(results,axis=0))
        print(np.std(results,axis=0))



