#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:06:35 2019

@author: manzars
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(iris.data, columns = iris.feature_names)
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['species'] = encoder.fit_transform(data['species'])

X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = "gini", random_state = 0)
classifier.fit(X_train, y_train)

y_pred = iris.target_names[classifier.predict(X_test)]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#abc
