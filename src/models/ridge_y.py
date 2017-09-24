"""
TODO:
"""

import sys
import tensorflow as tf

sys.path.append('../')


class RidgeY(object):
    """
    Ridge regression from label to query ( || x - Ry || + l*||R|| )
    """
    def __init__(self, **kwargs):
        self.d_x = kwargs.pop('d_x')
        self.d_y = kwargs.pop('d_y')
        self.l = kwargs.pop('l')
        self.l1 = kwargs.pop('l1')
        self._setup_cg()
        self._setup_scorer()

    def _setup_cg(self):
        self.train_x = tf.placeholder(tf.float32, [None, self.d_x])
        self.train_y = tf.placeholder(tf.float32, [None, self.d_y])

        n_train = tf.shape(self.train_x)[0]
        _l = tf.multiply(self.l, tf.cast(n_train, tf.float32))
        _l1 = tf.multiply(self.l1, tf.cast(n_train, tf.float32))

        with tf.name_scope('params'):
            initializer = tf.contrib.layers.xavier_initializer()
            self.r = tf.get_variable(name='r', shape=(self.d_y, self.d_x), initializer=initializer)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum((self.train_x - tf.matmul(self.train_y, self.r))**2) + _l * tf.nn.l2_loss(self.r) + _l1 * tf.reduce_sum(tf.abs(self.r))

    def _setup_scorer(self):
        self.test_x = tf.placeholder(tf.float32, [None, self.d_x])
        self.test_y = tf.placeholder(tf.float32, [None, self.d_y])

        n_test_x = tf.shape(self.test_x)[0]
        n_test_y = tf.shape(self.test_y)[0]

        with tf.name_scope('score'):
            query = tf.expand_dims(self.test_x, axis=1)
            label = tf.tile(tf.expand_dims(tf.matmul(self.test_y, self.r), axis=0), (n_test_x, 1, 1))
            self.score = tf.reduce_sum((query - label)**2, axis=2)  # shape = (n_test_x, n_test_y)
