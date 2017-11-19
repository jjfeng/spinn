import csv
import numpy as np

THRES = 1e-7
VAL_PROP_1FOLD = 0.8
ALMOST_ZERO = 0 #1e-8

def make_params(param_np_array):
    return ",".join(map(str, param_np_array))

def make_log_file_name(args):
    if min(args.group_lasso_params) > 0 or min(args.lasso_param_ratios) > 0:
        return "%s/log_%d_%d_s%s.txt" % (args.outdir, args.n_train, args.num_p, args.seeds.replace(",", "_"))
    else:
        return "%s/log_ridge_%d_%d_s%s.txt" % (args.outdir, args.n_train, args.num_p, args.seeds.replace(",", "_"))

def make_model_file_name(args, seed):
    if min(args.group_lasso_params) > 0 or min(args.lasso_param_ratios) > 0:
        fmt_str = "%s/model_%d_%d_%d.pkl"
    else:
        fmt_str = "%s/model_ridge_%d_%d_%d.pkl"
    return fmt_str % (args.outdir, args.n_train, args.num_p, seed)

def make_data_file_name(args, seed):
    if min(args.group_lasso_params) > 0 and args.num_p > 6:
        return "%s/data_%d_%d_%d.pkl" % (args.outdir, args.n_train, args.num_p, seed)
    else:
        return None

def process_params(param_str, dtype):
    if param_str:
        return [dtype(r) for r in param_str.split(",")]
    else:
        return []

def read_data(csv_filename, has_header=False):
    X = []
    y = []
    with open(csv_filename, "r") as f:
        csv_reader = csv.reader(f)
        if has_header:
            header = csv_reader.next()
        for row in csv_reader:
            X.append(row[:-1])
            y.append(row[-1])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    return X, y

def get_classification_accuracy(y1, y2):
    if y2.shape[1] == 1:
        return np.mean((y1 > 0.5) == (y2 > 0.5))
    else:
        return np.mean(
            np.argmax(y1, axis=1) == np.argmax(y2, axis=1)
        )

def get_regress_loss(y1, y2):
    return 0.5 * np.mean(np.power(y1 - y2, 2))

def get_summary(test_losses):
    mean = np.mean(test_losses)
    std_error = np.sqrt(np.var(test_losses)/len(test_losses))
    return "%f (%f)" % (mean, std_error)
