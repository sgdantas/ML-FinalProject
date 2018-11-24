"""
gcn
"""
from utils import load_data, preprocess_features, construct_feed_dict, preprocess_adj
from gcn.gcn_model import GCNN
import tensorflow as tf
import numpy as np
from settings import set_tf_flags, graph_settings
import time

SEED = 12
dataset_str = "cora"  # or citeseer
VERBOSE_TRAINING = True

flags = tf.app.flags
FLAGS = flags.FLAGS
settings = graph_settings()['default']
set_tf_flags(settings['params'], flags)

# Load the data/labels/adjacency matrix
adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)

print(features.shape)  # all the features
print(labels.shape)  # labels with every label of the dataset
print(adj.shape)  # Adjacency matrix

# Preprocess the features
features = preprocess_features(features)
# Set random seed
tf.set_random_seed(SEED)
np.random.seed(SEED)

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

support = preprocess_adj(adj)
# Create model
model = GCNN(placeholders, input_dim=features[2][1])

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, sub_sampled_support, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, sub_sampled_support, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, support, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # Validation
    cost, acc, duration, _ = evaluate(features, support, y_val, val_mask, support, placeholders)
    cost_val.append(cost)
    if VERBOSE_TRAINING:
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "train_acc=", "{:.5f}".format(
            outs[2]), "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc), "time=",
              "{:.5f}".format(time.time() - t))

    if FLAGS.early_stopping is not None and epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
            cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration, predicted_labels = evaluate(features, support, y_test, test_mask,
                                                                support, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc), "time=",
      "{:.5f}".format(test_duration))
labels_equal = (np.equal(np.argmax(predicted_labels, axis=1), np.argmax(y_test, axis=1)))
list_node_correctly_classified = np.argwhere(labels_equal).reshape(-1)
list_node_correctly_classified_test = list(filter(lambda x: test_mask[x], list(list_node_correctly_classified)))
tf.reset_default_graph()
