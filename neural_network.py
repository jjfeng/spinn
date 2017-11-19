import time
import logging as log
from parallel_worker import ParallelWorker
from parallel_worker import MultiprocessingManager

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from common import THRES, ALMOST_ZERO, get_classification_accuracy

from neural_network_wrapper import NeuralNetworkParams

class NeuralNetwork:
    """
    Right now this only uses tanh for middle layers and identity for output layer
    """
    @staticmethod
    def get_init_rand_bound_tanh(shape):
        # Used for tanh
        # Use the initialization method recommended by Glorot et al.
        return np.sqrt(6. / np.sum(shape))

    @staticmethod
    def get_init_rand_bound_sigmoid(shape):
        # Use the initialization method recommended by Glorot et al.
        return np.sqrt(2. / np.sum(shape))

    @staticmethod
    def create_tf_var(shape):
        # bound = NeuralNetwork.get_init_rand_bound_sigmoid(shape)
        bound = NeuralNetwork.get_init_rand_bound_tanh(shape)
        return tf.Variable(tf.random_uniform(shape, minval=-bound, maxval=bound))

    def __init__(self, settings):
        """
        @param regress: if False, then classify
        @param nn_classes: num classes, if zero, then regression
        """
        self.nn_classes = settings.data_classes
        if self.nn_classes < 2:
            num_out = 1
        else:
            num_out = self.nn_classes

        self.num_nonsmooth_layers = 1 # hard code that only the first layer is nonsmooth

        # Make tensorflow computation graph
        self.layer_sizes = [settings.num_p] + settings.hidden_sizes
        self.ridge_param = settings.ridge_param
        self.lasso_param = settings.lasso_param
        self.group_lasso_param = settings.group_lasso_param

        self.init_learn_rate = settings.learn_rate

        self.max_iters = settings.max_iters

        self.x = tf.placeholder(tf.float32, [None, self.layer_sizes[0]])
        self.y_true = tf.placeholder(tf.float32, [None, num_out])

        self.coefs = []
        self.coef_sizes = []
        self.intercepts = []
        self.intercept_sizes = []
        self.layers = []
        input_layer = self.x
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            W_size = [fan_in, fan_out]
            b_size = [fan_out]
            W = NeuralNetwork.create_tf_var(W_size)
            b = NeuralNetwork.create_tf_var(b_size)
            hidden_layer = tf.nn.tanh(tf.add(tf.matmul(input_layer, W), b))
            # hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, W), b))

            self.coef_sizes.append(W_size)
            self.intercept_sizes.append(b_size)
            self.coefs.append(W)
            self.intercepts.append(b)
            self.layers.append(hidden_layer)
            input_layer = hidden_layer

        # Make final layer
        W_out_size = [self.layer_sizes[-1], num_out]
        b_out_size = [num_out]
        W_out = NeuralNetwork.create_tf_var(W_out_size)
        b_out = NeuralNetwork.create_tf_var(b_out_size)
        self.coefs.append(W_out)
        self.coef_sizes.append(W_out_size)
        self.intercepts.append(b_out)
        self.intercept_sizes.append(b_out_size)
        if self.nn_classes == 0:
            self.y_pred = tf.add(tf.matmul(input_layer, W_out), b_out)
            self.loss = 0.5 * tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        elif self.nn_classes == 1:
            self.y_pred = tf.sigmoid(tf.add(tf.matmul(input_layer, W_out), b_out))
            self.loss = -tf.reduce_mean(tf.add(
                tf.multiply(self.y_true, tf.log(self.y_pred)),
                tf.multiply(1 - self.y_true, tf.log(1 - self.y_pred))
            ))
        else:
            self.y_pred = tf.add(tf.matmul(input_layer, W_out), b_out)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=self.y_pred))

        self.ridge_reg = tf.add_n([tf.nn.l2_loss(w) for w in self.coefs[1:]])
        if self.group_lasso_param == 0:
            # Actually, if we don't penalize the first layer at all, we should use a ridge penalty
            self.ridge_reg = tf.add_n([tf.nn.l2_loss(w) for w in self.coefs])
        self.smooth_pen_loss = tf.add(self.loss, 0.5 * self.ridge_param * self.ridge_reg)

        self.l1_reg = tf.reduce_sum(tf.abs(self.coefs[0])) # penalize only the first layer
        # self.l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.coefs])
        self.l2_reg = tf.reduce_sum(
            tf.reduce_sum(tf.pow(self.coefs[0], 2), axis=1)
        )
        self.all_pen_loss = tf.add(self.smooth_pen_loss, self.lasso_param * self.l1_reg + self.group_lasso_param * self.l2_reg)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        grad_optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        self.smooth_train_step = grad_optimizer.minimize(self.smooth_pen_loss)

        # Create gradient update placeholders and such
        self.var_list = self.coefs + self.intercepts
        self.var_sizes_list = self.coef_sizes + self.intercept_sizes
        self.smooth_pen_grad = grad_optimizer.compute_gradients(
            self.smooth_pen_loss,
            var_list=self.var_list,
        )
        self.placeholders = []
        self.assign_ops = []
        for v, v_size in zip(self.var_list, self.var_sizes_list):
            ph = tf.placeholder(
                tf.float32,
                shape=v_size,
            )
            assign_op = v.assign(ph)
            self.placeholders.append(ph)
            self.assign_ops.append(assign_op)

        self.scaler = StandardScaler()

    def update_penalty_parameters(self, settings):
        self.ridge_param = settings.ridge_param
        self.lasso_param = settings.lasso_param
        self.group_lasso_param = settings.group_lasso_param

        self.smooth_pen_loss = tf.add(self.loss, 0.5 * self.ridge_param * self.ridge_reg)
        self.all_pen_loss = tf.add(self.smooth_pen_loss, self.lasso_param * self.l1_reg + self.group_lasso_param * self.l2_reg)

    def fit_scaler(self, dataset):
        self.scaler.fit(dataset.x_train)

    def fit(self, dataset, print_iter, thres=1e-5, incr_thres=1.05, min_learning_rate=1e-5):
        st_time = time.time()
        # Use proximal gradient descent
        self.fit_scaler(dataset)
        x_train = self.scaler.transform(dataset.x_train)
        if dataset.x_val is not None:
            x_val = self.scaler.transform(dataset.x_val)
            y_val = dataset.y_val#_true
        else:
            x_val = None
            y_val = None
            self.val_loss = 0

        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            learn_rate = self.init_learn_rate
            prev_val = None
            prev_train = None
            self.all_pen_train_err = sess.run(
                [self.all_pen_loss],
                feed_dict={self.x: x_train, self.y_true: dataset.y_train}
            )
            for i in range(self.max_iters):
                # Do smooth gradient step
                smooth_pen_grad = sess.run(
                    self.smooth_pen_grad,
                    feed_dict={self.x: x_train, self.y_true: dataset.y_train, self.learning_rate: learn_rate}
                )
                potential_vars = []
                for var_list_idx, v in enumerate(self.var_list):
                    grad = smooth_pen_grad[var_list_idx][0]
                    var = smooth_pen_grad[var_list_idx][1]
                    update_val = var - learn_rate * grad
                    potential_vars.append(update_val)
                    if var_list_idx > 0:
                        sess.run(self.assign_ops[var_list_idx], feed_dict={self.placeholders[var_list_idx] : update_val})

                # Do proximal gradient step
                updated_coefs = []
                input_coef_val = potential_vars[0]
                if self.group_lasso_param > 0:
                    # Do proximal gradient step for sparse group lasso (where lasso param can be zero)
                    update_vals = []
                    # A group is an input node
                    for node_idx in range(self.layer_sizes[0]):
                        grouped_coefs = input_coef_val[node_idx,:]
                        if np.linalg.norm(grouped_coefs) < ALMOST_ZERO:
                            # If zero already, let it be.
                            group_coef_val_updated = grouped_coefs
                        else:
                            # soft threshold and then soft scale
                            group_coef_val_updated = np.multiply(
                                np.sign(grouped_coefs), np.maximum(np.abs(grouped_coefs) - self.lasso_param * learn_rate, ALMOST_ZERO)
                            )
                            if np.linalg.norm(group_coef_val_updated) > ALMOST_ZERO:
                                group_coef_val_updated = np.multiply(
                                    np.maximum(1 - self.group_lasso_param * learn_rate / np.linalg.norm(group_coef_val_updated), ALMOST_ZERO),
                                    group_coef_val_updated,
                                )

                        update_vals.append(group_coef_val_updated)
                    input_coef_val_updated = np.vstack(update_vals)
                    sess.run(self.assign_ops[0], feed_dict={self.placeholders[0] : input_coef_val_updated})
                    updated_coefs.append(input_coef_val_updated)
                else:
                    # Do proximal gradient step for lasso only
                    input_coef_val_updated = np.multiply(
                        np.sign(input_coef_val), np.maximum(np.abs(input_coef_val) - self.lasso_param * learn_rate, ALMOST_ZERO)
                    )
                    sess.run(self.assign_ops[0], feed_dict={self.placeholders[0] : input_coef_val_updated})
                    updated_coefs.append(input_coef_val_updated)

                self.all_pen_train_err = sess.run(
                    self.all_pen_loss,
                    feed_dict={self.x: x_train, self.y_true: dataset.y_train}
                )
                if prev_train is not None and self.all_pen_train_err > prev_train:
                    learn_rate /= 2

                    # Reset parameters to old values since the training loss went up instead
                    for var_list_idx, v in enumerate(self.var_list):
                        var = smooth_pen_grad[var_list_idx][1]
                        sess.run(self.assign_ops[var_list_idx], feed_dict={self.placeholders[var_list_idx] : var})

                    log.info("WENT UP: Training error went up: %f > %f" % (self.all_pen_train_err, prev_train))
                    continue
                else:
                    prev_train = self.all_pen_train_err

                if i % print_iter == 0 or i == self.max_iters - 1 or learn_rate < min_learning_rate:
                    if x_val is not None:
                        y_pred, self.val_loss = sess.run([self.y_pred, self.loss], feed_dict={self.x: x_val, self.y_true: y_val})
                        if self.nn_classes != 0:
                            log.info("  classification accur: %f" % get_classification_accuracy(y_val, y_pred))
                    log.info("Iter %d: Train error %f, val error %f (time %s)" % (i, self.all_pen_train_err, self.val_loss, str(time.time() - st_time)))

                    all_zero = False # If nn becomes all zero, stop training
                    for l_idx in range(len(updated_coefs)):
                        num_nonzero = self.layer_sizes[l_idx] - np.sum(np.max(np.abs(updated_coefs[l_idx]), axis=1) < THRES)
                        all_zero |= (num_nonzero == 0)
                        log.info("  layer %d, num nonzero %d" % (l_idx, num_nonzero))
                        if num_nonzero < 100:
                            nonzero_per_hidden = np.sum(np.abs(updated_coefs[l_idx]) >= THRES, axis=0)
                            nonzero_hidden_mask = nonzero_per_hidden > 0
                            log.info("    num nonzero into hidden node %s" % nonzero_per_hidden)

                            if np.sum(nonzero_hidden_mask) > 0:
                                log.info("    nonzero input nodes %s" % np.where(np.max(np.abs(updated_coefs[l_idx][:, nonzero_hidden_mask]), axis=1) > THRES))
                                nonzero_per_input = np.sum(np.abs(updated_coefs[l_idx][:, nonzero_hidden_mask]) > THRES, axis=1)

                                nonzero_inputs = np.where(nonzero_per_input)[0]
                                norm_nonzero_inputs = [
                                    np.linalg.norm(updated_coefs[l_idx][input_idx, nonzero_hidden_mask], ord=1)
                                    for input_idx in nonzero_inputs
                                ]

                                log.info("    num nonzero out from input node %s" % nonzero_per_input[np.where(nonzero_per_input)])
                                log.info("    weight norms out from input node %s" % norm_nonzero_inputs)


                    if all_zero:
                        break
                    # elif prev_val is not None and self.val_loss > incr_thres * prev_val:
                    #     # We won't be stopping early since our theory doesn't discuss it?
                    #     # log.info("STOP: Val loss increasing!")
                    #     # break
                    prev_val = self.val_loss

                if learn_rate < min_learning_rate:
                    log.info("not changing fast enough.")
                    break

            self.model_params = NeuralNetworkParams(
                [c.eval() for c in self.coefs],
                [b.eval() for b in self.intercepts]
            )
        sess.close()

    def _init_network_variables(self, sess):
        for i, best_c in enumerate(self.model_params.coefs):
            assign_op = self.coefs[i].assign(best_c)
            sess.run(assign_op)
        for i, best_b in enumerate(self.model_params.intercepts):
            assign_op = self.intercepts[i].assign(best_b)
            sess.run(assign_op)

    def predict(self, x, y):
        x_scaled = self.scaler.transform(x)
        sess = tf.Session()
        with sess.as_default():
            self._init_network_variables(sess)

            y_pred = sess.run(self.y_pred, feed_dict={self.x: x_scaled, self.y_true: y})
        sess.close()
        return y_pred
