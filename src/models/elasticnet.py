"""
TODO:
"""


import numpy as np
from sklearn.linear_model import ElasticNet


class ElasticNetX(ElasticNet):
    """
    Linear regression from query to label with L1 and L2 regularizers
    args:
      - alpha: coefficient value multiplied to regularizer
      - l1_ratio:
    """
    def __init__(self, **kwargs):
        self.d_x = kwargs.pop('d_x')
        self.d_y = kwargs.pop('d_y')
        self.l1_ratio = kwargs.pop('l1_ratio')
        self.alpha = kwargs.pop('alpha')
        super(ElasticNetX, self).__init__(alpha=self.alpha, l1_ratio=self.l1_ratio)

    def solve(self, train_x, train_y):
        self.fit(train_x, train_y)

    def cal_score(self, test_x, test_y):
        query = self.predict(test_x)
        label = test_y
        return np.array([[np.sum((l-q)**2) for l in label] for q in query])

    def get_param(self):
        return self.coef_


class ElasticNetY(ElasticNet):
    """
    Linear regression from label to query with L1 and L2 regularizers
    args:
      - alpha: coefficient value multiplied to regularizer
      - l1_ratio:
    """
    def __init__(self, **kwargs):
        self.d_x = kwargs.pop('d_x')
        self.d_y = kwargs.pop('d_y')
        self.l1_ratio = kwargs.pop('l1_ratio')
        self.alpha = kwargs.pop('alpha')
        super(ElasticNetY, self).__init__(alpha=self.alpha, l1_ratio=self.l1_ratio)

    def solve(self, train_x, train_y):
        self.fit(train_y, train_x)

    def cal_score(self, test_x, test_y):
        query = test_x
        label = self.predict(test_y)
        return np.array([[np.sum((l-q)**2) for l in label] for q in query])

    def get_param(self):
        return self.coef_
