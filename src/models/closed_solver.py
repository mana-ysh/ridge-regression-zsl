
import numpy as np


class ClosedSolver(object):
    def __init__(self):
        pass

    def solve(self):
        raise NotImplementedError

    def cal_score(self):
        raise NotImplementedError


class RidgeX(ClosedSolver):
    """
    Ridge regression from query to label ( || Rx - y || + l*||R|| )
    """
    def __init__(self, **kwargs):
        self.d_x = kwargs.pop('d_x')
        self.d_y = kwargs.pop('d_y')
        self.l = kwargs.pop('l')
        self.params = {'r': None}

    def solve(self, train_x, train_y):
        n_data = train_x.shape[0]
        _opt_r = train_y.T.dot(train_x).dot(np.linalg.inv(train_x.T.dot(train_x) + self.l * n_data * np.eye(train_x.shape[1])))
        self.params['r'] = _opt_r.T

    def cal_score(self, test_x, test_y):
        query = np.matmul(test_x, self.params['r'])
        label = test_y
        return np.array([[np.sum((l-q)**2) for l in label] for q in query])


class RidgeY(ClosedSolver):
    """
    Ridge regression from label to query ( || x - Ry || + l*||R|| )
    """
    def __init__(self, **kwargs):
        self.d_x = kwargs.pop('d_x')
        self.d_y = kwargs.pop('d_y')
        self.l = kwargs.pop('l')
        self.params = {'r': None}

    def solve(self, train_x, train_y):
        n_data = train_x.shape[0]
        _opt_r = train_x.T.dot(train_y).dot(np.linalg.inv(train_y.T.dot(train_y) + self.l * n_data * np.eye(train_y.shape[1])))
        self.params['r'] = _opt_r.T

    def cal_score(self, test_x, test_y):
        query = test_x
        label = np.matmul(test_y, self.params['r'])
        return np.array([[np.sum((l-q)**2) for l in label] for q in query])
