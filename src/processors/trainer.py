
import tensorflow as tf
import time


class SimpleTrainer(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop('model')
        self.epoch = kwargs.pop('epoch')
        self.sess = kwargs.pop('sess')
        self.opt = kwargs.pop('opt')
        self.logger = kwargs.pop('logger')

    """ Full batch training """
    def fit(self, train_x, train_y):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        grads_and_vars = self.opt.compute_gradients(self.model.loss)
        train_op = self.opt.apply_gradients(grads_and_vars, global_step=global_step)
        saver = tf.train.Saver(max_to_keep=0)  # saving models in every epoch
        self.sess.run(tf.initialize_all_variables())

        best_epoch = -1
        best_loss = float('inf')
        for epoch in range(self.epoch):
            start = time.time()
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            feed_dict = {self.model.train_x: train_x,
                         self.model.train_y: train_y}
            _, step, sum_loss = self.sess.run([train_op, global_step, self.model.loss], feed_dict)
            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

            if sum_loss < best_loss:
                best_epoch = epoch+1
                best_loss = sum_loss
        self.logger.info('===== Best loss: {} ({} epoch) ====='.format(best_loss, best_epoch))
