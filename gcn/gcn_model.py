from gcn.layers import *
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class GCNN(object):

    def __init__(self, placeholders, input_dim, hidden=None):
        self.vars = {}
        self.placeholders = {}
        self.logging = True
        self.name = 'gcn'
        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.hidden = hidden
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.inputs = placeholders['features']
        self.input_dim = input_dim

        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])

    def build(self):
        if self.hidden is None:
            self.hidden = FLAGS.hidden1
        with tf.variable_scope(self.name):
            self.layers.append(
                GraphConvolution(
                    input_dim=self.input_dim,
                    output_dim=self.hidden,
                    placeholders=self.placeholders,
                    support=self.placeholders['adjacency'],
                    act=tf.nn.relu,
                    dropout=True,
                    sparse_inputs=True,
                    logging=self.logging))

            self.layers.append(
                GraphConvolution(
                    input_dim=self.hidden,
                    output_dim=self.output_dim,
                    placeholders=self.placeholders,
                    support=self.placeholders['masked_adjacency'],
                    act=lambda x: x,
                    dropout=True,
                    logging=self.logging))

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        return tf.nn.softmax(self.outputs)
