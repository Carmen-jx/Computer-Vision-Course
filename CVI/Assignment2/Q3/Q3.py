import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
print("Loading MNIST data...")
train_df = pd.read_csv('Q3/mnist_train.csv')
test_df = pd.read_csv('Q3/mnist_test.csv')

#extractfeatures and labels and normalize pixel values
X_train = train_df.iloc[:, 1:].values / 255.0
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values / 255.0
y_test = test_df.iloc[:, 0].values

#==================
# MLP with Keras
#==================
print("\nTraining Keras MLP")
# Convert labels to categorical format
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

#define the MLP model
mlp_model = Sequential([
    Input(shape=(784,)), # 784 pixels for MNIST
    Dense(128, activation='relu'), # hidden layer with 128 neurons
    Dense(64, activation='relu'), # hidden layer with 64 neurons
    Dense(10, activation='softmax') # output layer for 10 classes (digits 0-9)
])

# compile the model with optimizer and loss function, and evaluation metric
mlp_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# fit the model on training data
mlp_model.fit(X_train, y_train_cat, epochs=10, batch_size=128, validation_split=0.1, verbose=2)

# evaluate the MLP model on test data
mlp_loss, mlp_acc = mlp_model.evaluate(X_test, y_test_cat, verbose=0)
#make predictions and class with highest probability
mlp_predictions = np.argmax(mlp_model.predict(X_test), axis=1)

#display MLP results
print(f"\n[MLP] Test Accuracy: {mlp_acc:.4f}")
print(classification_report(y_test, mlp_predictions))

#=============================
# K-Nearest Neighbors (KNN)
#=============================
print("\nTraining KNN (on subset)")
# create KNN model with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# train on a smaller subset for faster training
knn.fit(X_train[:5000], y_train[:5000]) 
knn_predictions = knn.predict(X_test[:1000]) # predict on a subset of test data

#print KNN results
print("[KNN] Accuracy:", accuracy_score(y_test[:1000], knn_predictions))
print(classification_report(y_test[:1000], knn_predictions))

#=============================
# Support Vector Machine (SVM)
#=============================
print("\nTraining SVM (on subset)")
# create SVM model with RBF kernel
svm = SVC(kernel='rbf', gamma='scale')

# train on a smaller subset for faster training (5000 samples)
svm.fit(X_train[:5000], y_train[:5000]) 
# predict on a subset of test data (1000 samples)
svm_predictions = svm.predict(X_test[:1000])

# print SVM results
print("[SVM] Accuracy:", accuracy_score(y_test[:1000], svm_predictions))
print(classification_report(y_test[:1000], svm_predictions))

#=============================
# Random Forest
#=============================
print("\nTraining Random Forest")
# create Random Forest model with 100 trees
rf = RandomForestClassifier(n_estimators=100)

# train on a smaller subset for faster training (10000 samples)
rf.fit(X_train[:10000], y_train[:10000])

# predict on the full test set
rf_predictions = rf.predict(X_test)

# print Random Forest results
print("[Random Forest] Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

#=============================
# Logistic Regression
#=============================
print("\nTraining Logistic Regression")
# create Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# train on a smaller subset for faster training (10000 samples)
log_reg.fit(X_train[:10000], y_train[:10000])

# predict on the full test set
log_predictions = log_reg.predict(X_test)

# print Logistic Regression results
print("[Logistic Regression] Accuracy:", accuracy_score(y_test, log_predictions))
print(classification_report(y_test, log_predictions))
