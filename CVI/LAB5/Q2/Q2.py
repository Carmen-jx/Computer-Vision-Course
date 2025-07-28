import pandas as pd                         
import numpy as np                          
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error  

#load the regression dataset (CSV format)
# this dataset contains population and benefit data
reg_df = pd.read_csv("data.csv")

#handle missing values in the 'population' column
# fill NaN values with the mean of the column
reg_df['population'] = reg_df['population'].fillna(reg_df['population'].mean())

#define the features and target variable for regression
# 'population' is the feature and 'benefit' is the target variable
X = reg_df[['population']]  # a 2D array
y = reg_df['benefit']       # a 1D array

# split the dataset into training and test sets (80/20 split)
# X will be the features and y will be the target variable
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize a StandardScaler to normalize the feature data
# this is important for regression to ensure the model performs well
lr = LinearRegression()
lr.fit(X_train_r, y_train_r)

# predict the target variable for the test set
# this will give us the predicted benefit values based on the population
y_pred_r = lr.predict(X_test_r)

# calculate error metrics:
mae = mean_absolute_error(y_test_r, y_pred_r)        
mse = mean_squared_error(y_test_r, y_pred_r)         
rmse = np.sqrt(mse)                                  

# print the results
print("\nQ2 - Linear Regression Results:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")