#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:47:28 2024

@author: markel
"""

import mglearn
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


X,y= mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=20)

sc=StandardScaler()
sc.fit(X_train)

X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn = Perceptron(
    max_iter=40,
    eta0=1,
    random_state=1
)
ppn.fit(X_train_std,y_train)

predicciones=ppn.predict(X_test_std)

print("Entradas mal clasificadas:%d"%(y_test !=predicciones).sum())

accuracy = accuracy_score(y_true = y_test, y_pred = predicciones, normalize = True)
accuracy=round(accuracy,2)

print(f"La exactitud del test es: {100*accuracy}%")

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

plot_decision_regions(X_combined_std, y_combined, ppn, X_highlight=X_test_std)