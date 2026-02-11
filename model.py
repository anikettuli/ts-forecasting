"""
Model training module - LightGBM with horizon-specific models
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from metrics import weighted_rmse_score


def train_model(
    train_df,
    valid_df,
    feature_cols,
    target_col="feature_ch",
    weight_col="feature_cg",
    num_leaves=127,
    max_depth=10,
    learning_rate=0.03,
):
    """
    Train a single LightGBM model.

    Args:
        train_df: Training dataframe
        valid_df: Validation dataframe
        feature_cols: List of feature columns
        target_col: Target column name
        weight_col: Weight column name
        num_leaves: LightGBM num_leaves parameter
        max_depth: LightGBM max_depth parameter
        learning_rate: LightGBM learning_rate parameter

    Returns:
        Trained model, validation predictions, validation score
    """
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col].values
    w_train = train_df[weight_col].fillna(1.0).values

    X_valid = valid_df[feature_cols].fillna(0)
    y_valid = valid_df[target_col].values
    w_valid = valid_df[weight_col].fillna(1.0).values

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    valid_data = lgb.Dataset(
        X_valid, label=y_valid, weight=w_valid, reference=train_data
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    preds = model.predict(X_valid)
    score = weighted_rmse_score(y_valid, preds, w_valid)

    return model, preds, score


def train_horizon_models(
    train_df, valid_df, feature_cols, target_col="feature_ch", weight_col="feature_cg"
):
    """
    Train separate models for each horizon.

    Args:
        train_df: Training dataframe
        valid_df: Validation dataframe
        feature_cols: List of feature columns
        target_col: Target column name
        weight_col: Weight column name

    Returns:
        Dictionary of models {horizon: model}, scores dict, overall score
    """
    models = {}
    scores = {}
    all_preds, all_true, all_weights = [], [], []

    horizons = sorted(train_df["horizon"].unique())

    for horizon in horizons:
        train_h = train_df[train_df["horizon"] == horizon]
        valid_h = valid_df[valid_df["horizon"] == horizon]

        if len(train_h) < 100 or len(valid_h) < 100:
            continue

        model, preds, score = train_model(
            train_h, valid_h, feature_cols, target_col, weight_col
        )

        models[horizon] = model
        scores[horizon] = score

        all_preds.extend(preds.tolist())
        all_true.extend(valid_h[target_col].values.tolist())
        all_weights.extend(valid_h[weight_col].fillna(1.0).values.tolist())

    overall = weighted_rmse_score(all_true, all_preds, all_weights)

    return models, scores, overall


def predict(df, models, feature_cols):
    """
    Generate predictions using horizon-specific models.

    Args:
        df: DataFrame to predict on
        models: Dictionary of {horizon: model}
        feature_cols: List of feature columns

    Returns:
        Array of predictions
    """
    predictions = np.zeros(len(df))

    for horizon, model in models.items():
        mask = df["horizon"] == horizon
        if mask.sum() > 0:
            X = df.loc[mask, feature_cols].fillna(0)
            predictions[mask] = model.predict(X)

    return predictions
