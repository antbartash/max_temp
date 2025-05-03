# Comparison of ML Methods for Time Series Modeling: CNNs, RNNs, and Traditional Approaches

This repository contains code for time series modeling using convolutional neural networks (CNNs), recurrent neural networks (RNNs), and traditional machine learning techniques (e.g., regression, gradient boosting).

The goal of the algorithms is to predict the maximum air temperature based on the temperatures from the previous 14 days and the day of the year.

Code for feature engineering for traditional ML models can be found in the data folder. The traditional_ml folder contains code for models built using the engineered features, while traditional_ml_w_base_features includes models that also use the raw time series values as additional features.

The mean absolute error (MAE) calculated on the validation dataset is presented below. The best results were achieved using CNNs and RNNs, although support vector regression (SVR) also performed well.


## traditional_ml

| Model               | MAE   | Hyperparameter tuning     | Candidates | CV-folds |
|--------------------|--------|--------------------------|------------|----------|
| KNN                 | 3.4062 | GridSearch               | 32         | 5        |
| RadiusNeighbors     | 3.4909 | Manual                   | 3          | -        |
| LinearRegression    | 3.1251 | GridSearch               | 250        | 5        |
| DecisionTree        | 3.2562 | GridSearch               | 1296       | 5        |
| RandomForest        | 3.1548 | BOGP, Multivariate TPE   | 100, 1000  | 3        |
| AdaBoost            | 3.1678 | Multivariate TPE         | 250        | 3        |
| XGBoost             | 3.1395 | Multivariate TPE         | 1000       | 3        |
| LightGBM            | 3.1224 | Multivariate TPE         | 1000       | 3        |
| CatBoost            | 3.1233 | Multivariate TPE         | 1000       | 3        |
| HistGradientBoosting| 3.1454 | Multivariate TPE         | 1000       | 3        |
| GradientBoosting    | 3.1455 | -                        | -          | -        |
| ExtraTrees          | 3.1434 | Multivariate TPE         | 839        | 3        |
| **LinearSVR**       | 3.0933 | GridSearch               | 22         | 5        |
| **SVR**             | 3.0529 | GridSearch               | 120        | 5        |
| VotingRegressor     | 3.0877 | Manual                   | -          | -        |
| StackingRegressor   | 3.0954 | Manual                   | 2          | -        |
| MLP                 | 3.0412 | TPE                      | 383        | -        |

## traditional_ml_w_base_features

| Model               | MAE   | Hyperparameter tuning     | Candidates | CV-folds |
|--------------------|--------|--------------------------|------------|----------|
| KNN                 | 3.4062 | GridSearch               | 32         | 5        |
| RadiusNeighbors     | 3.5856 | Manual                   | 3          | -        |
| LinearRegression    | 3.1236 | GridSearch               | 250        | 5        |
| DecisionTree        | 3.2308 | GridSearch               | 1296       | 5        |
| RandomForest        | 3.1501 | Multivariate TPE         | 362        | 3        |
| AdaBoost            | 3.1648 | Multivariate TPE         | 250        | 3        |
| XGBoost             | 3.1397 | Multivariate TPE         | 1000       | 3        |
| LightGBM            | 3.1241 | Multivariate TPE         | 1000       | 3        |
| CatBoost            | 3.1222 | Multivariate TPE         | 835        | 3        |
| HistGradientBoosting| 3.1361 | Multivariate TPE         | 1000       | 3        |
| GradientBoosting    | 3.1375 | -                        | -          | -        |
| ExtraTrees          | 3.1328 | Multivariate TPE         | 1000       | 3        |
| **LinearSVR**       | 3.0909 | GridSearch               | 22         | 5        |
| **SVR**             | 3.0473 | GridSearch               | 120        | 5        |
| VotingRegressor     | 3.0877 | Manual                   | -          | -        |
| StackingRegressor   | 3.0908 | Manual                   | 2          | -        |
| MLP                 | 3.0516 | TPE                      | 188        | -        |

## deep_learning

| Model               | MAE   | Hyperparameter tuning     | Candidates |
|--------------------|--------|--------------------------|------------|
| dense_baseline      | 2.9878 | Manual                   | 3          |
| **dense_tuned**     | 2.9908 | TPE                      | 72         |
| dense_tuned_manual  | 2.9821 | Manual                   | 7          |
| CNN_baseline        | 2.9742 | Manual                   | 3          |
| CNN_tuned           | 8.8032 | TPE                      | 47         |
| CNN_tuned_manual    | 2.9742 | Manual                   | 3          |
| SimpleRNN_baseline  | 2.9756 | Manual                   | 3          |
| **SimpleRNN_tuned** | 2.9736 | TPE                      | 242        |
| **LSTM_baseline**   | 2.9764 | Manual                   | 1          |
| LSTM_tuned          | 3.1852 | TPE                      | 11         |
| **GRU_baseline**    | 2.9768 | Manual                   | 3          |
| GRU_tuned           | 3.0154 | TPE                      | 12         |