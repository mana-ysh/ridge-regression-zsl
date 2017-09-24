"""
TODO
"""

import numpy as np


np.random.seed(46)


def gen_synthetic_data(d, d_x, d_y, n, skew_x=False, d_noise=0):
    assert d_x >= d_noise
    assert d_noise > -1
    d_info = d_x - d_noise
    z = np.random.normal(loc=0., scale=1.0, size=(n, d))
    r_x = np.random.uniform(-1, 1, size=(d, d_info))
    r_y = np.random.uniform(-1, 1, size=(d, d_y))
    x = z.dot(r_x) + np.random.randn(n, d_info)  # true_x + gaussian noise
    y = z.dot(r_y) + np.random.randn(n, d_y)
    if skew_x:
        eigenv = np.ones((d_x,))
        idx = np.random.randint(0, d_x)
        eigenv[idx] = 10.
        x = x.dot(np.diag(eigenv))
    if d_noise > 0:  # add noise feature (which is not correlated to label)
        noise_x = np.random.uniform(-1, 1, size=(n, d_noise))
        # noise_x = np.random.randn(n, d_noise)
        x = np.c_[x, noise_x]
    return x, y

if __name__ == '__main__':
    x, y = gen_synthetic_data(20, 100, 200, 10)
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)
