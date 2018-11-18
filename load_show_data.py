"""
Simple script to see the datasets/ how it is formatted, etc...
"""
from utils import load_data

dataset_str = "cora"  # or citeseer, pubmed. All citation ests used in Kipf paper

adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)

print(features.shape)  # all the features
print(labels.shape)  # labels with evrey label of the dataset
