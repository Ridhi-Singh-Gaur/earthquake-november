#!/usr/bin/env python
# coding: utf-8

# In this project we are predicting POSSIBILITY(in the form of 0 or 1) & the MAGNITUDE of AN EARTHQUAKE.
# Our model will give 1 as the outcome for earthquake will occur and 0 as the outcome for the earthquake will not occur.

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_curve, auc

# Load dataset
data = pd.read_csv("C://Users//ridhi//Downloads//query (1) (1).csv", encoding='latin-1')

# Removing unimportant columns
data = data.drop(["id", "updated", "status", "locationSource", "magSource",
                  "depthError", "magError", "magNst", "horizontalError", "type"], axis=1)

# Add earthquake column
data['earthquake'] = np.where(data['mag'] >= 3, 1, 0)

# Fill missing values
data["nst"].fillna(data["nst"].mean(), inplace=True)
data["gap"].fillna(data["gap"].mean(), inplace=True)
data["dmin"].fillna(data["dmin"].mean(), inplace=True)

# Date handling
data['time'] = pd.to_datetime(data['time'])
data["month"] = data["time"].dt.month
data = data.drop("time", axis=1)

# Extract place names
data['place'] = data['place'].apply(lambda x: x.split(', ')[1] if ', ' in x else x)

# Reorder columns
data = data[['month', 'latitude', 'longitude', 'depth', 'magType', 'nst', 'gap',
             'dmin', 'rms', 'net', 'place', 'earthquake', 'mag']]

# Label encoding
label_encoders = {}
categorical_columns = ['net', 'magType', 'place']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Prepare features and labels
X = data.iloc[:, :-2].values
Y = data.iloc[:, -2].values

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=11)
X_new = selector.fit_transform(X, Y)

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X_new)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(scaled_features, Y, test_size=1/3, random_state=0)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'K Nearest Neighbour': KNeighborsClassifier()
}

# Train and evaluate
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    results[name] = accuracy
    print(f"{name} → Accuracy: {accuracy*100:.2f}%")

# Find best classifier
best_classifier = max(results, key=results.get)
print("\n✅ Best classifier:", best_classifier, "with accuracy:", results[best_classifier])

# Save best model using pickle
import pickle

print("\n💾 Now saving the best model...")
best_model = classifiers[best_classifier]
best_model.fit(X_train, Y_train)

with open('earthquake_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("✅ Model trained & saved successfully as earthquake_model.pkl")

# Regression part (Magnitude prediction)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Feature selection for regression
selector1 = SelectKBest(f_classif, k=12)
x_new = selector1.fit_transform(x, y)

# Feature scaling
scaler1 = StandardScaler()
scaled_features1 = scaler1.fit_transform(x_new)

# Train linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Evaluate regression model
y_predictAll = regressor.predict(x)
print("\n📈 R-squared:", r2_score(y, y_predictAll) * 100)
print("📉 Mean Squared Error:", mean_squared_error(y, y_predictAll))
print("✅ Regression Model Score:", regressor.score(x_test, y_test) * 100)

print("\n🎉 Script executed successfully. Model file created → earthquake_model.pkl")
