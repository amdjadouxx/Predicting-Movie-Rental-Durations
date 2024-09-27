# Predicting Movie Rental Durations

## Project Overview
This project aims to help a DVD rental company predict how many days a customer will rent a DVD. The objective is to build a regression model that achieves a Mean Squared Error (MSE) of 3 or less. Such a model will assist the company in optimizing inventory management and rental planning.

## Dataset
The dataset `rental_info.csv` contains various features related to DVD rentals.

## Methodology
### 1. Data Preprocessing
The first step involves preprocessing the data by creating new features and removing unnecessary columns:
- **rental_length_days**: Calculated as the difference between `rental_date` and `return_date`, representing the number of days a DVD was rented.
- **Special Features Encoding**: Binary columns for whether the DVD includes deleted scenes or behind-the-scenes content.
- **Feature Removal**: Columns such as `special_features`, `rental_date`, and `return_date` are dropped as they are no longer needed.

### 2. Feature Selection with Lasso Regression
Lasso regression is used for feature selection, which helps identify the most relevant features by penalizing less important ones:
- **Lasso** applies a regularization term that forces some feature coefficients to zero, effectively removing them.
- After applying Lasso, the remaining features are used to train a **Linear Regression** model.

### 3. Model Training
Two different regression models were explored:
- **Linear Regression**: This is a basic regression model trained using the features selected by Lasso. It is evaluated on the test set to check its performance.
- **Random Forest Regressor**: A more complex ensemble method, which uses multiple decision trees to make predictions. A **RandomizedSearchCV** is applied to find the best hyperparameters for the model, such as the number of trees (`n_estimators`) and the maximum depth of the trees (`max_depth`).

### 4. Evaluation Metrics
The models are evaluated based on their **Mean Squared Error (MSE)**, which measures the average squared difference between the predicted and actual values. The goal is to minimize this error, aiming for an MSE of 3 or less.

## Conclusion
The Random Forest model, tuned with optimal hyperparameters, provided the best performance with an MSE of approximately **X**. This model will help the DVD rental company make better predictions about rental durations, allowing for more efficient inventory management.
