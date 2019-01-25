import time
import logging as log

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from neural_network_params import NeuralNetworkParams
from common import THRES, ALMOST_ZERO, get_classification_accuracy


class NeuralNetwork(BaseEstimator):
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

    def __init__(
            self,
            layer_sizes=None,
            data_classes=0,
            lasso_param_ratio=0.1,
            group_lasso_param=1,
            ridge_param=0.1,
            max_iters=1,
            num_inits=1,
            init_learn_rate=0.1,
            adam_learn_rate=0.001,
            adam_epsilon=1e-08,
            is_relu=0):
        self.data_classes = int(data_classes)

        self.num_nonsmooth_layers = 1 # hard code that only the first layer is nonsmooth

        # Make tensorflow computation graph
        self.layer_sizes = layer_sizes
        self.ridge_param = ridge_param
        self.lasso_param_ratio = lasso_param_ratio
        self.group_lasso_param = group_lasso_param
        self.init_learn_rate = init_learn_rate
        self.adam_learn_rate = adam_learn_rate
        self.adam_epsilon = adam_epsilon
        self.num_inits = int(num_inits)
        self.max_iters = int(max_iters)
        self.is_relu = is_relu
        if layer_sizes is not None:
            self._init_nn()

    def _init_nn(self):
        self.lasso_param = self.lasso_param_ratio * self.group_lasso_param
        if self.data_classes < 2:
            num_out = 1
        else:
            num_out = self.data_classes

        self.x = tf.placeholder(tf.float32, [None, self.layer_sizes[0]])
        self.y = tf.placeholder(tf.float32, [None, num_out])

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
            if self.is_relu:
                hidden_layer = tf.nn.relu(tf.add(tf.matmul(input_layer, W), b))
            else:
                hidden_layer = tf.nn.tanh(tf.add(tf.matmul(input_layer, W), b))

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
        if self.data_classes == 0:
            self.y_pred = tf.add(tf.matmul(input_layer, W_out), b_out)
            self.loss = 0.5 * tf.reduce_mean(tf.pow(self.y - self.y_pred, 2))
        elif self.data_classes == 1:
            self.y_pred = tf.sigmoid(tf.add(tf.matmul(input_layer, W_out), b_out))
            self.loss = -tf.reduce_mean(tf.add(
                tf.multiply(self.y, tf.log(self.y_pred)),
                tf.multiply(1 - self.y, tf.log(1 - self.y_pred))
            ))
        else:
            self.y_pred = tf.add(tf.matmul(input_layer, W_out), b_out)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))

        self.ridge_reg = tf.add_n([tf.nn.l2_loss(w) for w in self.coefs[1:]]) if len(self.coefs) > 1 else 0
        if self.group_lasso_param + self.lasso_param == 0:
            # Actually, if we don't penalize the first layer at all, we should use a ridge penalty
            self.ridge_reg = tf.add_n([tf.nn.l2_loss(w) for w in self.coefs])
        self.smooth_pen_loss = tf.add(self.loss, 0.5 * self.ridge_param * self.ridge_reg)

        self.l1_reg = tf.reduce_sum(tf.abs(self.coefs[0])) # penalize only the first layer
        self.l2_reg = tf.reduce_sum(
            tf.sqrt(tf.reduce_sum(tf.pow(self.coefs[0], 2), axis=1))
        )
        self.all_pen_loss = tf.add(self.smooth_pen_loss, self.lasso_param * self.l1_reg + self.group_lasso_param * self.l2_reg)

        self.grad_optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_learn_rate, epsilon=self.adam_epsilon)

        # Create gradient update placeholders and such
        self.var_list = self.coefs + self.intercepts
        self.var_sizes_list = self.coef_sizes + self.intercept_sizes

        self.smooth_train_step = self.grad_optimizer.minimize(self.smooth_pen_loss, var_list=self.var_list)
        #self.all_train_step = self.grad_optimizer.minimize(self.all_pen_loss, var_list=self.var_list)

        self.beta1_power, self.beta2_power = self.grad_optimizer._get_beta_accumulators()
        self.coef0_slot = self.grad_optimizer.get_slot(self.coefs[0], "v")
        self.smooth_pen_grad = self.grad_optimizer.compute_gradients(
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
        self.all_pen_grad = self.grad_optimizer.compute_gradients(
            self.all_pen_loss,
            var_list=self.var_list,
        )

    def fit(self, X, y):
        st_time = time.time()

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        x_scaled = self.scaler.transform(X)

        sess = tf.Session()
        best_loss = None
        self.model_params = None
        with sess.as_default():
            for n_init in range(self.num_inits):
                log.info("FIT INIT %d", n_init)
                tf.global_variables_initializer().run()
                try:
                    if self.group_lasso_param + self.lasso_param > 0:
                        #train_loss = self._fit_one_init_adam_prox(sess, x_scaled, y, self.max_iters)
                        train_loss = self._fit_one_init_prox(sess, x_scaled, y, self.max_iters)
                    else:
                        train_loss = self._fit_one_init_subgrad(sess, x_scaled, y, self.max_iters)
                    model_params = NeuralNetworkParams(
                        [c.eval() for c in self.coefs],
                        [b.eval() for b in self.intercepts],
                        self.scaler
                    )
                except AssertionError as e:
                    log.info("Assert error %s", str(e))
                    train_loss = None
                    model_params = None
                if train_loss is not None and (self.model_params is None or train_loss < best_loss):
                    self.model_params = model_params
                    best_loss = train_loss

        if best_loss is None:
            log.info("Couldn't fit the model well")
        else:
            log.info("FINAL best_loss %f (train time %f)", best_loss, time.time() - st_time)
            log.info("num nonzeros %s", self.model_params.nonzero_first_layer[0])
        log.info("layers %s", self.layer_sizes)
        log.info("lasso %f, group lasso %f", self.lasso_param, self.group_lasso_param)
        sess.close()

    def _fit_one_init_subgrad(self, sess, X, y, max_iters, print_iter=1000, thres=1e-5, incr_thres=1.05):
        log.info("ADAM begins")
        prev_val = None
        prev_train = None
        unpen_loss, all_pen_train_err = sess.run(
            [
                self.loss,
                self.smooth_pen_loss],
            feed_dict={self.x: X, self.y: y}
        )
        for i in range(max_iters):
            _, unpen_loss, all_pen_train_err = sess.run(
                [
                    self.smooth_train_step,
                    self.loss,
                    self.smooth_pen_loss],
                feed_dict={self.x: X, self.y: y}
            )
            if i % print_iter == 0 or i == self.max_iters - 1:
                log.info("Iter %d, unpen loss %f, loss %f", i, unpen_loss, all_pen_train_err)
            assert not np.isnan(all_pen_train_err)

        return all_pen_train_err

    def _fit_one_init_adam_prox(self, sess, X, y, max_iters, print_iter=1000, thres=1e-5, incr_thres=1.05, min_learning_rate=1e-8):
        log.info("ADAM+PROX begins")
        learn_rate = self.init_learn_rate
        prev_val = None
        prev_train = None
        unpen_loss, all_pen_train_err, coef0_slot, beta1_power, beta2_power = sess.run(
            [
                self.loss,
                self.all_pen_loss,
                self.coef0_slot,
                self.beta1_power,
                self.beta2_power],
            feed_dict={self.x: X, self.y: y}
        )
        print(beta1_power, beta2_power)
        for i in range(max_iters):
            _, unpen_loss, all_pen_train_err, coef0_slot, beta1_power, beta2_power = sess.run(
                [
                    self.smooth_train_step,
                    self.loss,
                    self.all_pen_loss,
                    self.coef0_slot,
                    self.beta1_power,
                    self.beta2_power],
                feed_dict={self.x: X, self.y: y}
            )
            #print(coef0_slot)
            lr = self.adam_learn_rate * np.sqrt(1 - beta2_power) / (1 - beta1_power)
            v_sqrt = np.sqrt(coef0_slot)
            adam_weight = lr / (v_sqrt + self.adam_epsilon)
            # Do proximal gradient step
            input_coef_val = self.coefs[0].eval()
            # Do proxmal gradient step for lasso: soft threshold
            input_coef_val_updated = np.multiply(
                np.sign(input_coef_val), np.maximum(np.abs(input_coef_val) - self.lasso_param * adam_weight, ALMOST_ZERO)
            )
            if self.group_lasso_param > 0:
                # Do proximal gradient step for group lasso: soft scale
                group_norms = 1e-10 + np.linalg.norm(input_coef_val_updated, axis=1).reshape(-1, 1)
                group_lasso_scale_factor = np.maximum(1 - self.group_lasso_param * adam_weight / group_norms, ALMOST_ZERO)
                input_coef_val_updated = np.multiply(group_lasso_scale_factor, input_coef_val_updated)

            updated_coefs = []
            if self.lasso_param + self.group_lasso_param > 0:
                sess.run(self.assign_ops[0], feed_dict={self.placeholders[0] : input_coef_val_updated})
                updated_coefs = [input_coef_val_updated]

            unpen_loss, all_pen_train_err = sess.run(
                [self.loss, self.all_pen_loss],
                feed_dict={self.x: X, self.y: y}
            )
            prev_train = all_pen_train_err

            if i % print_iter == 0 or i == self.max_iters - 1:
                log.info("Iter %d, unpen loss %f, loss %f", i, unpen_loss, all_pen_train_err)
                all_zero = False # If nn becomes all zero, stop training
                for l_idx in range(len(updated_coefs)):
                    num_nonzero = self.layer_sizes[l_idx] - np.sum(np.max(np.abs(updated_coefs[l_idx]), axis=1) < THRES)
                    all_zero |= (num_nonzero == 0)
                if all_zero:
                    log.info("ALL ZERO")
                    break

                for l_idx in range(len(updated_coefs)):
                    num_nonzero = self.layer_sizes[l_idx] - np.sum(np.max(np.abs(updated_coefs[l_idx]), axis=1) < THRES)
                    log.info("  layer %d, num nonzero %d" % (l_idx, num_nonzero))
                    nonzero_per_hidden = np.sum(np.abs(updated_coefs[l_idx]) >= THRES, axis=0)
                    nonzero_hidden_mask = nonzero_per_hidden > 0
                    log.info("    num nonzero into hidden node %s" % nonzero_per_hidden)
                    if num_nonzero < 100:
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

            if learn_rate < min_learning_rate:
                log.info("not changing fast enough.")
                break
        return all_pen_train_err

    def _fit_one_init_prox(self, sess, X, y, max_iters, print_iter=1000, thres=1e-5, incr_thres=1.05, min_learning_rate=1e-8):
        log.info("PROX begins")
        learn_rate = self.init_learn_rate
        prev_val = None
        prev_train = None
        unpen_loss, all_pen_train_err = sess.run(
            [
                self.loss,
                self.all_pen_loss],
            feed_dict={self.x: X, self.y: y}
        )
        for i in range(max_iters):
            # Do smooth gradient step
            smooth_pen_grad = sess.run(
                self.smooth_pen_grad,
                feed_dict={self.x: X, self.y: y}
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
            input_coef_val = potential_vars[0]
            # Do proxmal gradient step for lasso: soft threshold
            input_coef_val_updated = np.multiply(
                np.sign(input_coef_val), np.maximum(np.abs(input_coef_val) - self.lasso_param * learn_rate, ALMOST_ZERO)
            )
            if self.group_lasso_param > 0:
                # Do proximal gradient step for group lasso: soft scale
                group_norms = 1e-10 + np.linalg.norm(input_coef_val_updated, axis=1)
                group_lasso_scale_factor = np.maximum(1 - self.group_lasso_param * learn_rate / group_norms, ALMOST_ZERO).reshape(-1, 1)
                input_coef_val_updated = np.multiply(group_lasso_scale_factor, input_coef_val_updated)

            updated_coefs = []
            if self.lasso_param + self.group_lasso_param > 0:
                sess.run(self.assign_ops[0], feed_dict={self.placeholders[0] : input_coef_val_updated})
                updated_coefs = [input_coef_val_updated]

            unpen_loss, all_pen_train_err = sess.run(
                [self.loss, self.all_pen_loss],
                feed_dict={self.x: X, self.y: y}
            )
            if prev_train is not None and all_pen_train_err > prev_train:
                learn_rate *= 0.8

                # Reset parameters to old values since the training loss went up instead
                for var_list_idx, v in enumerate(self.var_list):
                    var = smooth_pen_grad[var_list_idx][1]
                    sess.run(self.assign_ops[var_list_idx], feed_dict={self.placeholders[var_list_idx] : var})

                log.info("WENT UP: Training error went up: %f > %f" % (all_pen_train_err, prev_train))
                continue
            else:
                prev_train = all_pen_train_err

            if i % print_iter == 0 or i == self.max_iters - 1 or learn_rate < min_learning_rate:
                log.info("Iter %d, unpen loss %f, loss %f", i, unpen_loss, all_pen_train_err)
                all_zero = False # If nn becomes all zero, stop training
                for l_idx in range(len(updated_coefs)):
                    num_nonzero = self.layer_sizes[l_idx] - np.sum(np.max(np.abs(updated_coefs[l_idx]), axis=1) < THRES)
                    all_zero |= (num_nonzero == 0)
                if all_zero:
                    log.info("ALL ZERO")
                    break

                for l_idx in range(len(updated_coefs)):
                    num_nonzero = self.layer_sizes[l_idx] - np.sum(np.max(np.abs(updated_coefs[l_idx]), axis=1) < THRES)
                    log.info("  layer %d, num nonzero %d" % (l_idx, num_nonzero))
                    nonzero_per_hidden = np.sum(np.abs(updated_coefs[l_idx]) >= THRES, axis=0)
                    nonzero_hidden_mask = nonzero_per_hidden > 0
                    log.info("    num nonzero into hidden node %s" % nonzero_per_hidden)
                    if num_nonzero < 100:
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

            if learn_rate < min_learning_rate:
                log.info("not changing fast enough.")
                break
        return all_pen_train_err

    def _init_network_variables(self, sess):
        for i, best_c in enumerate(self.model_params.coefs):
            assign_op = self.coefs[i].assign(best_c)
            sess.run(assign_op)
        for i, best_b in enumerate(self.model_params.intercepts):
            assign_op = self.intercepts[i].assign(best_b)
            sess.run(assign_op)

    def predict(self, x):
        x_scaled = self.scaler.transform(x)
        sess = tf.Session()
        with sess.as_default():
            self._init_network_variables(sess)

            y_pred = sess.run(self.y_pred, feed_dict={self.x: x_scaled})
        sess.close()
        return y_pred

    def score(self, x, y):
        if self.model_params is None:
            return -np.inf

        x_scaled = self.scaler.transform(x)
        sess = tf.Session()
        with sess.as_default():
            self._init_network_variables(sess)

            loss = sess.run(self.loss, feed_dict={self.x: x_scaled, self.y: y})
        sess.close()
        return -loss

    def get_params(self, deep=True):
        return {
            "layer_sizes": self.layer_sizes,
            "data_classes": self.data_classes,
            "lasso_param_ratio": self.lasso_param_ratio,
            "group_lasso_param": self.group_lasso_param,
            "ridge_param": self.ridge_param,
            "max_iters": self.max_iters,
            "num_inits": self.num_inits,
            "init_learn_rate": self.init_learn_rate,
            "is_relu": self.is_relu}

    def set_params(self, **params):
        if "layer_sizes" in params:
            self.layer_sizes = params["layer_sizes"]
        if "data_classes" in params:
            self.data_classes = int(params["data_classes"])
        if "lasso_param_ratio" in params:
            self.lasso_param_ratio = params["lasso_param_ratio"]
        if "group_lasso_param" in params:
            self.group_lasso_param = params["group_lasso_param"]
        if "ridge_param" in params:
            self.ridge_param = params["ridge_param"]
        if "max_iters" in params:
            self.max_iters = int(params["max_iters"])
        if "num_inits" in params:
            self.num_inits = int(params["num_inits"])
        if "init_learn_rate" in params:
            self.init_learn_rate = params["init_learn_rate"]
        if "is_relu" in params:
            self.is_relu = params["is_relu"]
        self._init_nn()
