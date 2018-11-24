
import numpy as np
from scipy import stats
from scipy.sparse.base import issparse


def cal_skewness(dist_mat, k, metric):
    """
    This code is based on 'hub-toobox'
    args:
      - dist_mat (ndarray) : Distant (Similarity) matrix (n_query, n_target)
      - k (int) : Neighborhood size for `k`-occurence
      - metric ({'similarity' or 'distant'}) : whether dist_mat is distant or similarity
    """

    if issparse(dist_mat):
        raise NotImplementedError()

    if metric == 'distance':
        self_val = np.inf
        sort_order = 1
    elif metric == 'similarity':
        self_val = -np.inf
        sort_order = -1
    else:
        raise ValueError('Invalid metric: {}'.format(metric))

    dist_mat = dist_mat.copy()
    n_query, n_target = dist_mat.shape
    kbest_idxs = np.zeros((k, n_query), dtype=np.float32)

    # np.fill_diagonal(dist_mat, self_val)
    dist_mat[~np.isfinite(dist_mat)] = self_val

    for i in range(n_query):
        dists = dist_mat[i, :]
        # dists[i] = self_val
        dists[~np.isfinite(dists)] = self_val

        # randomize equal values for avoiding high hubness (see original code)
        rand_idxs = np.random.permutation(n_target)
        dists2 = dists[rand_idxs]
        rank_dists2 = np.argsort(dists2, axis=0)[::sort_order]
        kbest_idxs[:, i] = rand_idxs[rank_dists2[0:k]]

    n_k = np.bincount(kbest_idxs.astype(int).ravel())
    skewness = stats.skew(n_k)

    return skewness
