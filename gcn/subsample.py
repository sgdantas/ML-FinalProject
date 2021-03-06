import random
import time
import numpy as np
import copy
from itertools import compress
import scipy.sparse as sp
#random.seed(123)


#remove columns from adj matrix.
#Be carefull too not modify the initial complete support matrix
def get_masked_adj(adj, node_to_keep):
    dense_adj = adj.todense()
    masked_adj = np.zeros(dense_adj.shape)
    masked_adj[:, node_to_keep] = dense_adj[:, node_to_keep]
   
    return sp.csc_matrix(masked_adj)


# Return a train mask for label_percent of the trainig set.
# if maintain_label_balance, keep smallest number of labels per class in training set that respect the label_percent, except for 100 %
def get_train_mask(label_percent, y_train, train_index, maintain_label_balance=False):

    train_mask = np.zeros((y_train.shape[0]), dtype=bool)  # list of False

    if maintain_label_balance:
        ones_index = []
        for i in range(y_train.shape[1]):  # find the ones for each class
            ones_index.append(np.argwhere(y_train[:, i] > 0).reshape(-1))

        if label_percent < 100:
            smaller_num = min(
                int(len(l) * (label_percent / 100))
                for l in ones_index)  # find smaller number of ones per class that respect the % constraint

            for ones in ones_index:
                random_index = random.sample(list(ones), smaller_num)
                train_mask[random_index] = True  # set the same number of ones for each class, so the set is balanced
        else:
            for ones in ones_index:
                train_mask[ones] = True

    else:
        random_sampling_set_size = int((label_percent / 100) * train_index.shape[0])
        random_list = random.sample(range(train_index.shape[0]), random_sampling_set_size)
        train_mask[random_list] = True

    return train_mask


#returns a random list of indexes of the node to be kept at random.
def get_random_percent(num_nodes, percent):
    if percent > 100:
        print("This is not how percentage works.")
        exit()
    random_sampling_set_size = int((percent * num_nodes) / 100)
    return random.sample(range(num_nodes), random_sampling_set_size)


#returns a list of indexes for the mask
def get_list_from_mask(mask):
    return list(compress(range(len(mask)), mask))


# Set features of node that shouldn't be in the set to crazy things to make sure they are not in the gcnn
def modify_features_that_shouldnt_change_anything(features, note_to_keep):
    note_doesnt_exist = [x for x in range(features[2][0]) if x not in note_to_keep]
    a = np.where(np.isin(features[0][:, 0], note_doesnt_exist))
    features[1][a[0]] = 10000000
