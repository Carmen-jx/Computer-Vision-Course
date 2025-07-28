import pandas as pd                         
import numpy as np                          
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error  

# load the fruit dataset (tab-separated values)
fruit_df = pd.read_csv("fruit_data_with_colors.txt", sep='\t')

# select relevant features for KNN classification
features = fruit_df[['mass', 'width', 'height', 'color_score']]

# define the target variable (fruit name)
target = fruit_df['fruit_name']

# encode the target variable using LabelEncoder
# this converts categorical labels into numerical values
le = LabelEncoder()
target_encoded = le.fit_transform(target)

# split the dataset into training and test sets (80/20 split)
# X will be the features and y will be the encoded target variable
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

#initialize a StandardScaler to normalize the feature data
# this is important for KNN to ensure all features contribute equally
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)  # Fit the scaler on training data and transform it
X_test_norm = scaler.transform(X_test)        # transform the test data using the same scaler

# initialize a dictionary to store results
results = {'k': [], 'accuracy_raw': [], 'accuracy_normalized': []}

# loop through odd values of k from 1 to 20
# using odd values helps avoid ties in classification
for k in range(1, 21, 2):
    # train KNN on raw data
    # KNeighborsClassifier is used for classification tasks
    knn_raw = KNeighborsClassifier(n_neighbors=k)
    knn_raw.fit(X_train, y_train)
    y_pred_raw = knn_raw.predict(X_test)
    acc_raw = accuracy_score(y_test, y_pred_raw)  # evaluate accuracy

    # train KNN on normalized data
    # this ensures that the model is not biased towards features with larger scales
    knn_norm = KNeighborsClassifier(n_neighbors=k)
    knn_norm.fit(X_train_norm, y_train)
    y_pred_norm = knn_norm.predict(X_test_norm)
    acc_norm = accuracy_score(y_test, y_pred_norm)  # evaluate accuracy

    #. save the results for each k
    results['k'].append(k)
    results['accuracy_raw'].append(acc_raw)
    results['accuracy_normalized'].append(acc_norm)

# convert results to a DataFrame for better visualization
df_results = pd.DataFrame(results)

# print the results
print("Q1 - KNN Accuracy Comparison:\n", df_results)