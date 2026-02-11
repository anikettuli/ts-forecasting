---
active: true
iteration: 10
maxIterations: 100
sessionId: ses_3b4f65bf0ffe4ke9akR6T45jWa
---

Objective: Achieve a weighted_rmse_score of 0.90+ using data/test.parquet. Metric: SkillScore=1−∑wy2∑w(y−y^​)2​​
1. Environment & Constraints

    Source: data/test.parquet.

    Leakage Prevention: For any ts_index = t, features must only use information from 0…t.

    ID Structure: id is code__sub_code__sub_category__horizon__ts_index.

    Weight Handling: Use the weight column for the loss function, but never as a feature.

2. The Iterative Loop (Task List)

Perform these tasks sequentially. After each successful run (metric improvement), commit to git and clear the context to start the next iteration.

    Iteration A: The "Golden" Split & Metric

        Implement the weighted_rmse_score function provided in the competition description.

        Split the data into train and valid sets based on ts_index. Use the last 25% of the available time indices as the validation set to simulate the public leaderboard.

    Iteration B: Temporal Feature Engineering

        Group by code, sub_code, and sub_category.

        Calculate rolling averages and expanding means for all 86 features (feature_a to feature_ch).

        Crucial: Shift these features by 1 to ensure no leakage of the current index's value.

    Iteration C: The "Weighted" Model

        Train a LightGBM Regressor or XGBoost.

        Custom Loss: Pass the weight column into the sample_weight parameter of the fit function to align the model directly with the competition metric.

        Treat horizon as a categorical feature or train 4 separate models (one for each horizon: 1, 3, 10, 25).

    Iteration D: Signal-to-Noise Refinement

        The data has a "low signal-to-noise ratio." Implement a Denoising Autoencoder or a simple PCA on the 86 features to extract the strongest signals before feeding them to the GBM.

        Test if sub_category embeddings improve performance for codes with sparse data.

3. Submission Protocol

    The final output must be a CSV with id and prediction.

    Ensure the code runs in under 6 hours and processes ts_index sequentially.

Implementation Tip for 90% Accuracy:

The metric is effectively a Weighted R² variant. Because it divides by ∑wy2, the model is heavily penalized for missing high-magnitude targets.

    Pro-Tip: Focus your feature engineering on the relationship between code and sub_category. The prompt mentions similarities across these categories—use Target Encoding on the sub_code and sub_category columns, but calculate the means in a "moving window" fashion to prevent data leakage.