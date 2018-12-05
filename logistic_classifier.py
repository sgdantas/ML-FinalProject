# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:35:17 2018

@author: gadda
"""

from utils import load_data
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import KFold
kf = KFold(5, True, 1)

dataset_str = "citeseer"  # or citeseer, pubmed. All citation ests used in Kipf paper

adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)

labels_class = np.argmax(labels,axis = 1)
#%%
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(features, labels_class, test_size=0.33, random_state=42)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [400,600,800,1000],#'max_depth' : [4,8,12,20],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)

CV_rfc.best_params_
#%%
accuracy = []
for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels_class[train_index], labels_class[test_index]
        clf_rf = RandomForestClassifier(n_estimators=500,criterion='gini', max_depth=400,max_features='log2',random_state=42).fit(X_train, y_train)
        #rf_probs = clf_rf.predict_proba(X_test)
        accuracy.append(clf_rf.score(X_test, y_test))

#%%
accuracy_logistic = []
for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels_class[train_index], labels_class[test_index]
        clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
        accuracy_logistic.append(clf.score(X_test, y_test))
#%%
import statistics
print(statistics.stdev(accuracy_logistic))