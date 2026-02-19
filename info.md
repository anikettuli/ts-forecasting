# Hedge Fund - Time Series Forecasting: Master Guide

This document serves as your master guide for the Kaggle competition "Hedge fund - Time series forecasting" (TS-Forecasting Vol. 1).

## 1. Overview and Objective
The core objective of this competition is to train a machine learning model that generalized robustly out-of-sample to predict future values for specific combinations of identifiers across different time horizons.

*   **Target:** Predict a continuous numerical value for each combination of `code`, `sub_code`, `sub_category`, and `horizon` at a given `ts_index`.
*   **Key Challenge:** Ensure the model accurately captures temporal patterns while adhering to strict causality rules (i.e., NO future data usage).

## 2. Evaluation Metric
The competition uses a **weighted RMSE-based Skill Score** to measure performance relative to a naive benchmark.

*   **Metric Formula:** 
    $$Score = \sqrt{1 - \text{clipping}(\text{ratio})}$$
    where:
    $$\text{ratio} = \frac{\sum_{i \in I} w_i (y_i - \hat{y}_i)^2}{\sum_{i \in I} w_i y_i^2}$$
*   **Components:**
    *   $y_i$: The ground truth target value.
    *   $\hat{y}_i$: The predicted value.
    *   $w_i$: Weights associated with each prediction.
    *   **Clipping:** The ratio is clipped between 0 and 1. If your model is worse than the trivial benchmark (ratio > 1), your score will be 0.
*   **Public/Private Split:** 
    *   **Public Leaderboard:** Calculated on **25%** of the test data.
    *   **Private Leaderboard:** Calculated on **75%** of the test data (final ranking).

## 3. Data Description
The competition provides data in `.parquet` format for efficiently handling large time-series datasets.

### Datasets
*   `train.parquet`: Historical data used for training.
*   `test.parquet`: The test set for which predictions must be generated.
*   `sample_submission.csv`: A template for submission with `id` and `prediction` columns.

### Columns and Features
The dataset contains **86 anonymized features** (`feature_a` through `feature_ch`). 
*   **`code` / `sub_code` / `sub_category`:** Hierarchical categorical identifiers for the time series.
*   **`ts_index`:** A numerical index representing the time step (actual dates/times are hidden).
*   **`horizon`:** The forecast horizon (how far into the future the prediction is for).
*   **`target`:** The value to be predicted (only available in the training set).

## 4. Competition Rules and Guidelines
*   **No Data Leakage:** Forecasts for a specific `ts_index` must not use any information (features or targets) from any index greater than the forecast's `ts_index`.
*   **Submission Format:** While `.csv` submissions are supported, using Kaggle Notebooks is recommended for better reproducibility and to avoid submission size issues.
*   **Data Integrity:** Exact timestamps are hidden to prevent the use of external data (e.g., matching with real-world stock prices if the data were financial).

## 5. Prizes
Total Prize Pool: **$10,000 USD**
*   **1st Place:** $3,500
*   **2nd Place:** $2,500
*   **3rd Place:** $2,000
*   **4th Place:** $1,000
*   **5th Place:** $1,000
*   **Career Opportunity:** The top 5 participants are additionally offered the opportunity for job interviews with the hosting hedge fund.

## 6. Tips for Success
1.  **Feature Engineering:** Since features are anonymized, focus heavily on temporal transformations (lags, rolling means, exponential moving averages, differences) and interactions between the hierarchical identifiers.
2.  **Weighting ($w_i$):** Pay close attention to how the weights are structured in the evaluation metric; recent periods or specific categories may carry more weight and therefore should be prioritized by the model.
3.  **Cross-Validation:** Implement a rigorous time-series cross-validation strategy (such as an expanding window `TimeSeriesSplit` or embargoed splits) to ensure your model's stability over hidden, out-of-sample time steps without leaking future information.
