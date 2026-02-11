"""
Metrics module - Weighted RMSE calculation
"""

import numpy as np


def weighted_rmse_score(y_true, y_pred, weights):
    """
    Calculate weighted RMSE skill score.

    SkillScore = 1 - sqrt(sum(w * (y - y_hat)^2) / sum(w * y^2))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    weighted_squared_error = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y_squared = np.sum(weights * y_true**2)

    if weighted_y_squared == 0:
        return 0.0

    return 1 - np.sqrt(weighted_squared_error / weighted_y_squared)
