import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Step 1: Import necessary libraries and the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a logistic regression model with C = 1e4
model = LogisticRegression(C=1e4, max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model and report the classification accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Classification Accuracy: {accuracy:.4f}')