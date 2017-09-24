"""
TODO:
"""

import sys
import tensorflow as tf

sys.path.append('../')


class RidgeX(object):
    """
    Ridge regression from query to label ( || Rx - y || + l*||R|| )
    """
    def __init__(self, **kwargs):
        self.d_x = kwargs.pop('d_x')
        self.d_y = kwargs.pop('d_y')
        self.l = kwargs.pop('l')  # L2 reguralization
        self._setup_cg()
        self._setup_scorer()

    def _setup_cg(self):
        self.train_x = tf.placeholder(tf.float32, [None, self.d_x])
        self.train_y = tf.placeholder(tf.float32, [None, self.d_y])

        n_train = tf.shape(self.train_x)[0]
        _l = tf.multiply(self.l, tf.cast(n_train, tf.float32))

        with tf.name_scope('params'):
            initializer = tf.contrib.layers.xavier_initializer()
            self.r = tf.get_variable(name='r', shape=(self.d_x, self.d_y), initializer=initializer)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum((tf.matmul(self.train_x, self.r) - self.train_y)**2) + _l * tf.nn.l2_loss(self.r)

    def _setup_scorer(self):
        self.test_x = tf.placeholder(tf.float32, [None, self.d_x])
        self.test_y = tf.placeholder(tf.float32, [None, self.d_y])

        n_test_x = tf.shape(self.test_x)[0]
        n_test_y = tf.shape(self.test_y)[0]

        with tf.name_scope('score'):
            query = tf.expand_dims(tf.matmul(self.test_x, self.r), axis=1)
            label = tf.tile(tf.expand_dims(self.test_y, axis=0), (n_test_x, 1, 1))
            self.score = tf.reduce_sum((query - label)**2, axis=2)  # shape = (n_test_x, n_test_y)
