
import sys

sys.path.append('../')
from lib.hubness import cal_skewness
from utils.math_util import find_nearest_neighbor


class Evaluator(object):
    def __init__(self, mode, **kwargs):
        self.mode = mode
        if self.mode == 'online':
            self.sess = kwargs.pop('sess', None)
        elif self.mode == 'closed' or self.mode == 'cd':
            pass
        else:
            raise NotImplementedError('Invalid mode: {}'.format(self.mode))

    def set_sess(self, sess):
        self.sess = sess

    """computing accuracy and skewness@10 in object matching"""
    def run(self, model, test_x, test_y, gold_x2y=None):
        assert len(test_x) == len(test_y), 'Different number of samples'
        if not gold_x2y:
            print('=== CAUTION: assuming gold pairs are ordered ===')
            gold_x2y = list(range(len(test_x)))
        if self.mode == 'online':
            score_mat = self._score_online(model, test_x, test_y, gold_x2y)
        else:  # closed
            score_mat = self._score_closed(model, test_x, test_y, gold_x2y)
        nn_labes = find_nearest_neighbor(score_mat)
        n_corr = sum(1 for x_id, y_id in enumerate(nn_labes) if gold_x2y[x_id] == y_id)
        skewness = cal_skewness(score_mat, 10, 'distance')
        return n_corr / len(test_x), skewness

    """evaluation for tensorflow's model"""
    def _score_online(self, model, test_x, test_y, gold_x2y):
        feed_dict = {model.test_x: test_x, model.test_y: test_y}
        score_mat = self.sess.run(model.score, feed_dict)
        return score_mat

    """evaluation for closed form's model"""
    def _score_closed(self, model, test_x, test_y, gold_x2y):
        score_mat = model.cal_score(test_x, test_y)
        return score_mat
