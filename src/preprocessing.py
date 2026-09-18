"""Preprocessing utilities."""

import numpy as np


def create_sequences(data, target_index, input_width=72, horizon=24):
    X, y = [], []
    for i in range(input_width - 1, len(data) - horizon):
        X.append(data[i - input_width + 1:i + 1])
        y.append(data[i + horizon, target_index])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32).reshape(-1, 1),
    )


def standardize(train, validation, test):
    mean = train.mean()
    std = train.std()
    return (
        (train - mean) / std,
        (validation - mean) / std,
        (test - mean) / std,
        mean,
        std,
    )
