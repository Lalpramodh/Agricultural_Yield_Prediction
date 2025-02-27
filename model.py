# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ucn9ye8JmjhMrnuAvyZnpiBkgtFHY7ED
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("Crop_recommendation.csv")
print(df.head())

df.shape

df.isnull().sum()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

new_features = [[117 ,32,34,26.2724184,52.12739421,6.758792552,127.1752928,]]
predicted_crop = model.predict(new_features)
print("Predicted crop:", predicted_crop)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

from sklearn.tree import DecisionTreeClassifier
import pickle

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)