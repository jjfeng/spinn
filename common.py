import csv
import pickle
import numpy as np

THRES = 1e-7
ALMOST_ZERO = 0

def make_params(param_np_array):
    return ",".join(map(str, param_np_array))

def process_params(param_str, dtype):
    if param_str:
        return [dtype(r) for r in param_str.split(",")]
    else:
        return []

def read_input_data(args):
    import pandas as pd
    from data_generator import Dataset
    if args.data_index_file is None:
        with open(args.data_file, "rb") as f:
            all_data = pickle.load(f)
            train_data = all_data["train"]
            test_data = all_data["test"]
    else:
        with open(args.data_index_file, "rb") as f:
            data_indices = pickle.load(f)
            train_indices = data_indices["train"]
            test_indices = data_indices["test"]
            print(train_indices)
            print(test_indices)
        Xdata = pd.read_csv(args.data_X_file).values[:, 1:]
        rand_cols = np.random.choice(
            Xdata.shape[1],
            size=min(Xdata.shape[1], 5000),
            replace=False)
        Xdata = Xdata[:, rand_cols]
        print(Xdata)
        ydata = pd.read_csv(args.data_y_file).values
        print(ydata)
        train_data = Dataset(
            Xdata[train_indices, :],
            ydata[train_indices, :],
            ydata[train_indices, :])
        test_data = Dataset(
            Xdata[test_indices, :],
            ydata[test_indices, :],
            ydata[test_indices, :])
    return train_data, test_data

def write_data_pkl_to_csv(data_file, train_file_name, test_file_name):
    with open(data_file, "rb") as f:
        all_data = pickle.load(f)
        train_data = all_data["train"]
        test_data = all_data["test"]

    train_dat = np.hstack([train_data.x, train_data.y_true, train_data.y])
    np.savetxt(train_file_name, train_dat, delimiter=',')
    test_dat = np.hstack([test_data.x, test_data.y_true, test_data.y])
    np.savetxt(test_file_name, test_dat, delimiter=',')

def read_data(csv_filename, has_header=False):
    X = []
    y = []
    with open(csv_filename, "r") as f:
        csv_reader = csv.reader(f)
        if has_header:
            header = next(csv_reader)
        for row in csv_reader:
            X.append(row[:-1])
            y.append(row[-1])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    return X, y

def get_logistic_loss(y1, y2):
    if y2.shape[1] == 1:
        y1 = y1.ravel()
        y2 = y2.ravel() > 0.5
        # binary classification
        return np.mean(np.log(y1) * y2 + np.log(1 - y1) * (1 - y2))
    else:
        raise ValueError("not implemented")

def get_classification_accuracy(y1, y2):
    if y2.shape[1] == 1:
        y1 = y1.ravel()
        y2 = y2.ravel()
        # binary classification
        print(y1, y2)
        return np.mean((y1 > 0.5) == (y2 > 0.5))
    else:
        # multiclass
        return np.mean(
            np.argmax(y1, axis=1) == np.argmax(y2, axis=1)
        )

def get_regress_loss(y1, y2):
    y1 = y1.ravel()
    y2 = y2.ravel()
    return 0.5 * np.mean(np.power(y1 - y2, 2))

def get_summary(test_losses):
    mean = np.mean(test_losses)
    std_error = np.sqrt(np.var(test_losses)/len(test_losses))
    return "%f (%f)" % (mean, std_error)
