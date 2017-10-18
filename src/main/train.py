"""
Ridge regression between two domain object matching by using synthetic data
"""

import argparse
from datetime import datetime
import logging
import os
try:
    from sklearn.cross_validation import train_test_split
except:
    from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf

sys.path.append('../')
from processors.evaluator import Evaluator
from processors.trainer import *
from utils.dataset import gen_synthetic_data
from utils.math_util import *

DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '{}'.format(datetime.now().strftime('%Y%m%d_%H:%M')))


DIM = 3000
DIM_X = 300
DIM_Y = 300
NUM = 10000

def train(args):
    # setting for logging
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.log, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info('Arguments...')
    for arg, val in vars(args).items():
        logger.info('{:>10} -----> {}'.format(arg, val))

    if args.sk:
        from sklearn.datasets import make_regression
        x, y = make_regression(n_samples=NUM, n_features=DIM_X, n_targets=DIM_Y, n_informative=args.info, noise=1.)
    else:
        x, y = gen_synthetic_data(DIM, DIM_X, DIM_Y, NUM, args.skewx, DIM_X-args.info)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    if args.stdx:
        logger.info('standardize data...')
        sc = StandardScaler()
        sc.fit(train_x)
        train_x = sc.transform(train_x)
        test_x = sc.transform(test_x)
    evaluator = Evaluator(mode=args.mode)

    if args.mode == 'online':
        with tf.Graph().as_default():
            tf.set_random_seed(46)
            sess = tf.Session()
            opt = tf.train.AdamOptimizer() # TODO: make other optimizers available
            if args.method == 'ridgex':
                from models.ridge_x import RidgeX
                model = RidgeX(d_x=DIM_X, d_y=DIM_Y, l=args.l)
            elif args.method == 'ridgey':
                from models.ridge_y import RidgeY
                model = RidgeY(d_x=DIM_X, d_y=DIM_Y, l=args.l, l1=args.l1)
            else:
                raise NotImplementedError

            trainer = SimpleTrainer(model=model, epoch=args.epoch, opt=opt,
                                    sess=sess, logger=logger)
            trainer.fit(train_x, train_y)

            logger.info('evaluation...')
            evaluator.set_sess(sess)
            accuracy, skewness = evaluator.run(model, test_x, test_y)

    elif args.mode == 'closed':
        if args.method == 'ridgex':
            from models.closed_solver import RidgeX
            model = RidgeX(d_x=DIM_X, d_y=DIM_Y, l=args.l)
        elif args.method == 'ridgey':
            from models.closed_solver import RidgeY
            model = RidgeY(d_x=DIM_X, d_y=DIM_Y, l=args.l)
        else:
            raise NotImplementedError

        logger.info('calculating closed form solution...')
        model.solve(train_x, train_y)

        logger.info('evaluation...')
        accuracy, skewness = evaluator.run(model, test_x, test_y)

    elif args.mode == 'cd':
        if args.method == 'ridgex':
            from models.elasticnet import ElasticNetX
            model = ElasticNetX(d_x=DIM_X, d_y=DIM_Y, alpha=args.l, l1_ratio=args.l1_ratio)
        elif args.method == 'ridgey':
            from models.elasticnet import ElasticNetY
            model = ElasticNetY(d_x=DIM_X, d_y=DIM_Y, alpha=args.l, l1_ratio=args.l1_ratio)

        logger.info('calculating solution...')
        model.solve(train_x, train_y)
        print(model.get_param().tolist())

        logger.info('evaluation...')
        accuracy, skewness = evaluator.run(model, test_x, test_y)

    else:
        raise ValueError('Invalid mode: {}'.format(args.mode))

    logger.info('  accuracy  : {}'.format(accuracy))
    logger.info('skewness@10 : {}'.format(skewness))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='online', type=str, help='training mode ["online", "closed", "cd"]')
    p.add_argument('--method', default='ridgex', type=str, help='method ["ridgex", "ridgey"]')
    p.add_argument('--lr', default=0.1, type=float, help='learning rate')
    p.add_argument('--epoch', default=1000, type=int, help='number of epochs')
    p.add_argument('--log', default=DEFAULT_LOG_DIR, type=str, help='output log dir')
    p.add_argument('--l', default=0., type=float, help='regularizer')
    p.add_argument('--l1', default=0., type=float, help='L1 regularizer')
    p.add_argument('--l1_ratio', default=10e-10, type=float)
    p.add_argument('--skewx', action='store_true', help='force to skew x in synthetic data')
    p.add_argument('--stdx', action='store_true', help='standardize x in synthetic data')
    p.add_argument('--info', default=DIM_X, type=int, help='informative dimention of scikit-learn')
    p.add_argument('--sk', action='store_true', help='use scikit-learn dataset')
    p.add_argument('--abs', action='store_true', help='take abs in reg2')
    p.add_argument('--r2_flg', action='store_true', help='enable L2 reg for r2')

    train(p.parse_args())
