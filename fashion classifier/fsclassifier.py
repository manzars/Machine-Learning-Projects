#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:58:48 2019

@author: manzars
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

data = pd.read_csv("fashion_train.csv")

X = data.iloc[:, 1:].values
y = data.iloc[:, 0:1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

classifier = RandomForestClassifier(n_estimators = 500, n_jobs = 3, criterion = "gini", random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

def check_prediction(num):
    plt.gray()
    plt.matshow(X_test[num].reshape(28,28))
    plt.show()
    
    print("Predicted Fashion was %r" %(check_fashion(int(classifier.predict(X_test[num].reshape(1, -1))))))
    
    
def check_model_accuracy(cm):
    sum = 0
    for i in range(10):
        for j in range(10):
            if(i == j):
                sum = sum + cm[i][j]
    return sum / len(X_test) * 100



flag = True
while(flag):
    choice = int(input("1. Make prediction\n2. exit\n"))
    if(choice == 1):
        print("Model Accuracy was %s So there can be chances of Wromg prediction" %(check_model_accuracy(cm)))
        num = int(input("Enter the Index between (0, 8399) TO make Prediction: "))
        check_prediction(num)
    elif(choice == 2):
        flag = False
    else:
        print("Wrong Input please try again...")
        
def check_fashion(num):
    fashion = {0: "T-shirt/top",1: "Trouser",2: "Pullover",3: "Dress",4: "Coat",5: "Sandal",6: "Shirt",7: "Sneaker",8: "Bag",9: "Ankle boot "}
    return fashion[num]
