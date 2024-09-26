import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor

class MovieRentalDurationPredictor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.lasso_selected_features = self.select_features_with_lasso()
        self.linear_regression_model = self.train_linear_regression()
        self.random_forest_model, self.best_mse = self.train_random_forest()

    def preprocess_data(self):
        """Preprocess the data by creating new features and dropping unnecessary columns."""
        self.df["rental_length"] = pd.to_datetime(self.df["return_date"]) - pd.to_datetime(self.df["rental_date"])
        self.df["rental_length_days"] = self.df["rental_length"].dt.days
        self.df["deleted_scenes"] = np.where(self.df["special_features"].str.contains("Deleted Scenes"), 1, 0)
        self.df["behind_the_scenes"] = np.where(self.df["special_features"].str.contains("Behind the Scenes"), 1, 0)
        self.df.drop(columns=["special_features", "rental_length", "rental_date", "return_date"], inplace=True)

    def split_data(self):
        """Split the data into training and testing sets."""
        X = self.df.drop(columns=["rental_length_days"])
        y = self.df["rental_length_days"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def select_features_with_lasso(self):
        """Select features using Lasso regression."""
        lasso = Lasso(alpha=0.01)
        lasso.fit(self.X_train, self.y_train)
        return self.X_train.columns[lasso.coef_ != 0]

    def train_linear_regression(self):
        """Train a Linear Regression model using Lasso-selected features."""
        X_train_lasso = self.X_train[self.lasso_selected_features]
        X_test_lasso = self.X_test[self.lasso_selected_features]
        ols = LinearRegression()
        ols.fit(X_train_lasso, self.y_train)
        y_test_pred = ols.predict(X_test_lasso)
        mse_linreg_lasso = mean_squared_error(self.y_test, y_test_pred)
        print(f"Linear Regression MSE with Lasso-selected features: {mse_linreg_lasso}")
        return ols

    def train_random_forest(self):
        """Train a Random Forest model using hyperparameter tuning."""
        param_dist = {'n_estimators': np.arange(1, 101, 1), 'max_depth': np.arange(1, 11, 1)}
        rf = RandomForestRegressor(random_state=9)
        rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, cv=5, random_state=9)
        rand_search.fit(self.X_train, self.y_train)
        best_params = rand_search.best_params_

        rf_best = RandomForestRegressor(n_estimators=best_params["n_estimators"],
                                        max_depth=best_params["max_depth"],
                                        random_state=9)
        rf_best.fit(self.X_train, self.y_train)
        y_pred = rf_best.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Random Forest MSE: {mse}")
        return rf_best, mse

if __name__ == "__main__":
    predictor = MovieRentalDurationPredictor('rental_info.csv')