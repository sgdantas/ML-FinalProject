"""
gcn
"""
from utils import load_data, preprocess_features, construct_feed_dict, preprocess_adj
from gcn.gcn_model import GCNN
import tensorflow as tf
import numpy as np
from settings import set_tf_flags, graph_settings
import time
from sklearn.model_selection import StratifiedShuffleSplit
from gcn.subsample import get_train_mask

SEED = 125
NUM_CROSS_VAL = 10
dataset_str = "cora"  # or citeseer
VERBOSE_TRAINING = False
LABEL_TRAINING_PERCENT = 27

flags = tf.app.flags
FLAGS = flags.FLAGS
settings = graph_settings()['default']
set_tf_flags(settings['params'], flags)

# Set random seed
tf.set_random_seed(SEED)
np.random.seed(SEED)

# Load the data/labels/adjacency matrix
adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)

n = features.shape[0]
# Preprocess the features
features = preprocess_features(features)

test_split = StratifiedShuffleSplit(n_splits=NUM_CROSS_VAL, test_size=0.40, random_state=SEED)
test_split.get_n_splits(labels, labels)

results_cross_validation = np.zeros((NUM_CROSS_VAL,))
i = 0
for train_index, test_index in test_split.split(labels, labels):

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[train_index[0:1208]] = True
    val_mask[train_index[1208:]] = True
    test_mask[test_index] = True

    y_train = np.zeros(labels.shape, dtype=int)
    y_val = np.zeros(labels.shape, dtype=int)
    y_test = np.zeros(labels.shape, dtype=int)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # Semi supervised, hide a percentage of training labels

    train_mask = get_train_mask(LABEL_TRAINING_PERCENT, y_train, train_index[0:1208], maintain_label_balance=True)

    # Define placeholders
    placeholders = {
        'masked_adjacency': tf.sparse_placeholder(tf.float32),
        'adjacency': tf.sparse_placeholder(tf.float32),
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero':
            tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    adjacency = preprocess_adj(adj)
    # Create model
    model = GCNN(placeholders, input_dim=features[2][1])

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, adjacency, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, adjacency, labels, mask, adjacency, placeholders)
        outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, adjacency, y_train, train_mask, adjacency, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        # Validation
        cost, acc, duration, _ = evaluate(features, adjacency, y_val, val_mask, placeholders)
        cost_val.append(cost)
        if VERBOSE_TRAINING:
            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "train_acc=",
                  "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc),
                  "time=", "{:.5f}".format(time.time() - t))

        if FLAGS.early_stopping is not None and epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
                cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration, predicted_labels = evaluate(features, adjacency, y_test, test_mask,
                                                                    placeholders)
    print("Cross Val:", str(i + 1), "Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=",
          "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    labels_equal = (np.equal(np.argmax(predicted_labels, axis=1), np.argmax(y_test, axis=1)))
    list_node_correctly_classified = np.argwhere(labels_equal).reshape(-1)
    list_node_correctly_classified_test = list(filter(lambda x: test_mask[x], list(list_node_correctly_classified)))
    results_cross_validation[i] = test_acc
    i += 1
    tf.reset_default_graph()

print("Average Accuracy:", "{:.3f}".format(np.average(results_cross_validation)), "+/-", "{:.3f}".format(
    np.std(results_cross_validation)))
