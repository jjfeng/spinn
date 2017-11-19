class Settings:
    def __init__(self, nn_cls, func, n_train, n_val, n_test, num_p, snr, ridge_param, lasso_param, group_lasso_param, hidden_sizes, learn_rate=0.5, num_inits=2, max_iters=4000, data_classes=0):
        self.nn_cls = nn_cls
        self.func = func
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.num_p = num_p
        self.snr = snr
        self.ridge_param = ridge_param
        self.lasso_param = lasso_param
        self.group_lasso_param = group_lasso_param
        self.hidden_sizes = hidden_sizes
        self.learn_rate = learn_rate
        self.max_iters = max_iters
        self.num_inits = num_inits
        # if zero, then regression. else classification
        self.data_classes = data_classes

    def update(self, settings_mini):
        self.ridge_param = settings_mini.ridge_param
        self.lasso_param = settings_mini.lasso_param
        self.group_lasso_param = settings_mini.group_lasso_param
        self.hidden_sizes = settings_mini.hidden_sizes

class SettingsMini:
    def __init__(self, hidden_sizes, ridge_param, lasso_param=0, group_lasso_param=0):
        self.ridge_param = ridge_param
        self.lasso_param = lasso_param
        self.group_lasso_param = group_lasso_param
        self.hidden_sizes = hidden_sizes

    def __str__(self):
        return "h:%s, r:%.4e, l:%.4e, g:%.4e" % (self.hidden_sizes, self.ridge_param, self.lasso_param, self.group_lasso_param)
