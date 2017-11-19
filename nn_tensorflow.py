import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from sklearn.preprocessing import StandardScaler

from nonparam_lasso_multivariate import my_func_hard
from nonparam_lasso_multivariate import create_data
from settings import Settings

THRES = 1e-7

def get_init_rand_bound(shape):
    # Use the initialization method recommended by Glorot et al.
    return np.sqrt(6. / np.sum(shape))

def create_tf_var(shape):
    bound = get_init_rand_bound(shape)
    return tf.Variable(tf.random_uniform(shape, minval=-bound, maxval=bound))

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

# ha. lots of settings. equals settingss.
settingss = [
    Settings(
        my_func_hard,
        size_train = 1000,
        size_test = 2000,
        num_p = 50,
        snr = 2,
        ridge_param = 0.2,
        lasso_param = 0.02,
        group_lasso_param = 0,
        hidden_sizes=[5],
        learn_rate = 0.5,
        max_iters = 5001,
    ),
    Settings(
        my_func_hard,
        size_train = 4000,
        size_test = 2000,
        num_p = 50,
        snr = 2,
        ridge_param = 0.1,
        lasso_param = 0.01,
        group_lasso_param = 0,
        hidden_sizes=[5],
        learn_rate = 0.5,
        max_iters = 5001,
    ),
    Settings(
        my_func_hard,
        size_train = 10000,
        size_test = 2000,
        num_p = 50,
        snr = 2,
        ridge_param = 0.05,
        lasso_param = 0.005,
        group_lasso_param = 0,
        hidden_sizes=[5],
        learn_rate = 0.5,
        max_iters = 5001,
    ),
]
settings = settingss[2]

# Make tensorflow computation graph
x = tf.placeholder(tf.float32, [None, settings.num_p])
y_true = tf.placeholder(tf.float32, [None, 1])

W_size = [settings.num_p, settings.hidden_sizes[0]]
b_size = [settings.hidden_sizes[0]]
W = create_tf_var(W_size)
b = create_tf_var(b_size)

hidden_layer = tf.nn.tanh(tf.add(tf.matmul(x, W), b))

W_out_size = [settings.hidden_sizes[0], 1]
b_out_size = [1]
W_out = create_tf_var(W_out_size)
b_out = create_tf_var(b_out_size)

y_pred = tf.add(tf.matmul(hidden_layer, W_out), b_out)
loss = tf.reduce_mean(0.5 * tf.pow(y_true - y_pred, 2))
ridge_pen_loss = tf.add(loss, 0.5 * settings.ridge_param * tf.nn.l2_loss(W_out))
lasso_ridge_pen_loss = tf.add(ridge_pen_loss, settings.lasso_param * tf.reduce_sum(tf.abs(W)))
train_step = tf.train.GradientDescentOptimizer(settings.learn_rate).minimize(ridge_pen_loss)
ridge_pen_loss_grad = tf.train.GradientDescentOptimizer(settings.learn_rate).compute_gradients(
    ridge_pen_loss,
    var_list=[W, b, W_out, b_out]
)

variables = [W, b, W_out, b_out]
variable_sizes = [W_size, b_size, W_out_size, b_out_size]

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

x_train, y_train, y_train_true = create_data(
    settings.size_train,
    settings.num_p,
    true_func=settings.my_func,
    snr=settings.snr,
)
x_test, _, y_test_true = create_data(
    settings.size_test,
    settings.num_p,
    true_func=settings.my_func,
    snr=settings.snr,
)
y_train = np.reshape(y_train, (y_train.size, 1))
y_train_true = np.reshape(y_train_true, (y_train_true.size, 1))
y_test_true = np.reshape(y_test_true, (y_test_true.size, 1))

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

with sess.as_default():
    #### Use LBFGS instead
    # def obj_func(theta):
    #     last_idx = 0
    #     for var, var_shape in zip(variables, variable_sizes):
    #         var_size = np.prod(var_shape)
    #         var_idx_range = np.arange(last_idx, last_idx + var_size)
    #         var_val = np.reshape(theta[var_idx_range], var_shape)
    #         new_v = var.assign(var_val)
    #         sess.run(new_v)
    #         last_idx += var_size
    #
    #     ridge_pen_train_err = sess.run(ridge_pen_loss, feed_dict={x: x_train, y_true: y_train})
    #     return np.array(ridge_pen_train_err, dtype=float)
    #
    # def grad_func(theta):
    #     ridge_pen_train_err_grad = sess.run(ridge_pen_loss_grad, feed_dict={x: x_train, y_true: y_train})
    #     grad_flat = np.concatenate([v[0].flatten() for v in ridge_pen_train_err_grad])
    #
    #     return np.array(grad_flat, dtype=float)
    #
    # init_params = np.concatenate([v.eval().flatten() for v in variables])
    # init_loss = sess.run(ridge_pen_loss, feed_dict={x: x_train, y_true: y_train})
    # print "init_loss", init_loss
    # res = minimize(obj_func, init_params, method="L-BFGS-B", jac=grad_func, options={"maxiter": 10000, "disp": True})
    # print "success?", res.success
    # print "final loss", res.fun
    # test_loss = sess.run(loss, feed_dict={x: x_test, y_true: y_test_true})
    # print "test_loss (sqrt)", np.sqrt(test_loss)
    # train_loss = sess.run(loss, feed_dict={x: x_train, y_true: y_train_true})
    # print "train loss (sqrt)", np.sqrt(train_loss)

    #### Use (proximal) gradient descent
    for i in range(settings.max_iters):
        lasso_ridge_pen_train_err = sess.run(lasso_ridge_pen_loss, feed_dict={x: x_train, y_true: y_train})

        sess.run(train_step, feed_dict={x: x_train, y_true: y_train})
        W_val = W.eval()
        W_val_updated = np.multiply(np.sign(W_val), np.maximum(np.abs(W_val) - settings.lasso_param * settings.learn_rate, 0))
        new_W = W.assign(W_val_updated)
        sess.run(new_W)
        if i % 500 == 0:
            test_err = sess.run(loss, feed_dict={x: x_test, y_true: y_test_true})
            print("Iter %d: Train error %f, test erro %f" % (i, np.sqrt(lasso_ridge_pen_train_err), np.sqrt(test_err)))
            num_nonzero = settings.num_p - np.sum(np.max(np.abs(W_val_updated), axis=1) < THRES)
            print("  num nonzero %d" % num_nonzero)
            if num_nonzero < 15:
                print("  nonzero %s" % np.where(np.max(np.abs(W_val_updated), axis=1) > THRES))
