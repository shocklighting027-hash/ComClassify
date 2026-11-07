import numpy as np
import pandas as pd


def train_test_split(X, y, test_size=None, random_state=None, stratify=False):

    if not test_size:
        test_size = 0.3

    if random_state is None:
        random_state = 42

    if not stratify:
        rng = np.random.default_rng(seed=random_state)
        shuffle_data = rng.permutation(np.arange(X.shape[0]))
        train_slice = np.round(X.shape[0] * (1 - test_size)).astype(int)
        train_idx = shuffle_data[:train_slice]
        test_idx = shuffle_data[train_slice:]

        if hasattr(X, "iloc"):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        else:
            X_train, X_test = X[train_idx], X[test_idx]

        if hasattr(y, "iloc"):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        else:
            y_train, y_test = y[train_idx], y[test_idx]

    else:
        y_arr = np.asarray(y).ravel()
        classes = np.unique(y_arr)
        rng = np.random.default_rng(seed=random_state)
        train_indexes = np.array([], dtype="int32")
        test_indexes = np.array([], dtype="int32")

        for cls in classes:
            cls_data_y = np.where(y_arr == cls)[0]
            shuffle_data = rng.permutation(np.array(cls_data_y))
            train_slice = np.round(
                cls_data_y.size * (1 - test_size)).astype(int)
            train_indexes = np.concatenate(
                (train_indexes, shuffle_data[:train_slice]), dtype="int32"
            )
            test_indexes = np.concatenate(
                (test_indexes, shuffle_data[train_slice:]), dtype="int32"
            )

        rng.shuffle(train_indexes)
        rng.shuffle(test_indexes)

        if hasattr(X, "iloc"):
            X_train, X_test = X.iloc[train_indexes], X.iloc[test_indexes]

        else:
            X_train, X_test = X[train_indexes], X[test_indexes]

        if hasattr(y, "iloc"):
            y_train, y_test = y.iloc[train_indexes], y.iloc[test_indexes]

        else:
            y_train, y_test = y[train_indexes], y[test_indexes]

    return X_train, X_test, y_train, y_test
