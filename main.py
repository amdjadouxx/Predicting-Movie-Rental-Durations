import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("rental_info.csv")
df["rental_length"] = pd.to_datetime(df["return_date"]) - pd.to_datetime(df["rental_date"])
df["rental_length_days"] = df["rental_length"].dt.days
df["deleted_scenes"] = np.where(df["special_features"].str.contains("Deleted Scenes"), 1, 0)
df["behind_the_scenes"] = np.where(df["special_features"].str.contains("Behind the Scenes"), 1, 0)
cols_to_drop = ["special_features", "rental_length", "rental_length_days", "rental_date", "return_date"]
X = df.drop(cols_to_drop, axis=1)
y = df["rental_length_days"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

df["rental_length"] = pd.to_datetime(df["return_date"]) - pd.to_datetime(df["rental_date"])
df["rental_length_days"] = df["rental_length"].dt.days
df["deleted_scenes"] = np.where(df["special_features"].str.contains("Deleted Scenes"), 1, 0)
df["behind_the_scenes"] = np.where(df["special_features"].str.contains("Behind the Scenes"), 1, 0)
lasso = Lasso(alpha=0.3, random_state=9)
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_
X_lasso_train, X_lasso_test = X_train.iloc[:, lasso_coef > 0], X_test.iloc[:, lasso_coef > 0]

cols_to_drop = ["special_features", "rental_length", "rental_length_days", "rental_date", "return_date"]
X = df.drop(cols_to_drop, axis=1)
y = df["rental_length_days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

lasso = Lasso(alpha=0.3, random_state=9)
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_
X_lasso_train, X_lasso_test = X_train.iloc[:, lasso_coef > 0], X_test.iloc[:, lasso_coef > 0]

ols = LinearRegression()
ols = ols.fit(X_lasso_train, y_train)
y_test_pred = ols.predict(X_lasso_test)
mse_linreg_lasso = mean_squared_error(y_test, y_test_pred)

param_dist = {'n_estimators': np.arange(1,101,1),
          'max_depth':np.arange(1,11,1)}

rf = RandomForestRegressor()
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions=param_dist, 
                                 cv=5, 
                                 random_state=9)
rand_search.fit(X_train, y_train)
hyper_params = rand_search.best_params_

rf = RandomForestRegressor(n_estimators=hyper_params["n_estimators"],
                          max_depth=hyper_params["max_depth"],
                          random_state=9)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

best_model = rf
best_mse = mse
