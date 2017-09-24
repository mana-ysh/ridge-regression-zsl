
import numpy as np
import tensorflow as tf

def find_nearest_neighbor(dist_mat):
    """
    Args:
      - dist_mat (np.ndarray): distance matrix (shape = (n_query, n_label))
    Return:
      - nn_labels (list): label ids of query's nearest neighbor (shape = (n_query, ))
    """
    return [np.argmin(score) for score in dist_mat]


def eye_initializer(shape):
    init = np.eye(*shape)
    return tf.constant_initializer(init)
