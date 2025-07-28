import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error  # ✅ New import
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# load dataset  
df = pd.read_csv("Q1/house_price.csv")

# set up features and target variable
# 'size' and 'bedroom' are the features and 'price' is the target variable
X = df[['size', 'bedroom']]
y = df['price']
# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize the features
# for SGDRegressor to ensure all features contribute equally
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# use Linear Regression model to predict house prices
linear_model = LinearRegression()
linear_model.fit(X_train, y_train) # fit the model on the training data
y_pred_linear = linear_model.predict(X_test) # predict on the test data

# display the coefficients and intercept of the linear regression model
print("Linear Regression Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# use SGD Regressor model to predict house prices
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_train_scaled, y_train) # fit the model on the scaled training data
y_pred_sgd = sgd_model.predict(X_test_scaled) # predict on the scaled test data

#function to evaluate the model performance
def evaluate(y_true, y_pred): 
    mae = mean_absolute_error(y_true, y_pred) # calculate Mean Absolute Error
    mse = mean_squared_error(y_true, y_pred) # calculate Mean Squared Error
    rmse = np.sqrt(mse) # calculate Root Mean Squared Error
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100 # calculate Mean Absolute Percentage Error
    return mae, mse, rmse, mape

# evaluate the performance of both models
metrics_linear = evaluate(y_test, y_pred_linear)
metrics_sgd = evaluate(y_test, y_pred_sgd)

#display results
print("\n--- Model Performance Comparison ---")
print(f"Linear Regression - MAE: {metrics_linear[0]:.2f}, MSE: {metrics_linear[1]:.2f}, RMSE: {metrics_linear[2]:.2f}, MAPE: {metrics_linear[3]:.2f}%")
print(f"SGD Regressor     - MAE: {metrics_sgd[0]:.2f}, MSE: {metrics_sgd[1]:.2f}, RMSE: {metrics_sgd[2]:.2f}, MAPE: {metrics_sgd[3]:.2f}%")