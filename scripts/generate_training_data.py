from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import h5py

from lib.utils import MinMaxScaler

def generate_graph_seq2seq_io_data(
        X, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param X: pollution data
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = X.shape
    X = np.expand_dims(X, axis=-1)
    data_list = [X]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    X = np.concatenate(data_list, axis=-1)
    print(X.shape)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = X[t + x_offsets, ...]
        y_t = X[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    # get pollution data
    data_file = args.df_filename
    if os.path.isfile(data_file):
        with h5py.File(data_file, 'r') as hf:
            X = hf['pollution'][:]
    print(X.shape)

    # 0 is the latest observed sample.
    seq_len = args.seq_len
    x_offsets = np.sort(
        np.concatenate((np.arange(-(seq_len-1), 1, 1),))
    )
    # Predict the next one hour
    horizon = args.horizon
    y_offsets = np.sort(np.arange(1, (horizon+1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim), input_dim = input features' dimensions
    # y: (num_samples, output_length, num_nodes, output_dim), output_dim = output features' dimensions
    x, y = generate_graph_seq2seq_io_data(
        X, 
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    #num_test = round(num_samples * 0.2)
    #num_train = round(num_samples * 0.7)
    #num_val = num_samples - num_test - num_train
    num_train = 8760 + 8784 # 2 years data
    num_val = 92*24 # last 3 months
    num_train = num_train - num_val
    num_test = num_samples - num_train - num_val # remaning 1 year data

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--seq_len", type=int, default="12", help="Sequence length"
    )
    parser.add_argument(
        "--horizon", type=int, default="12", help="Horizon length"
    )
    parser.add_argument(
        "--df_filename",
        type=str,
        default="data/pm25_grid.h5",
        help="Pollution data readings.",
    )
    args = parser.parse_args()
    main(args)
