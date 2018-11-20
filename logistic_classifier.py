# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:35:17 2018

@author: gadda
"""

from utils import load_data
from sklearn.linear_model import LogisticRegression
import numpy as np

dataset_str = "cora"  # or citeseer, pubmed. All citation ests used in Kipf paper

adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)

#%%
from sklearn.model_selection import train_test_split
labels_class = np.argmax(labels,axis = 1)
x_train,x_test,y_train,y_test = train_test_split(features, labels_class, test_size=0.33, random_state=42)

#%%
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(x_train, y_train)
clf.score(x_test, y_test)

#%%
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0).fit(x_train, y_train)
clf_rf.score(x_test, y_test)