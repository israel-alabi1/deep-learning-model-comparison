"""Evaluation utilities."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def inverse_transform_target(values, target_mean, target_std):
    return np.asarray(values) * target_std + target_mean
