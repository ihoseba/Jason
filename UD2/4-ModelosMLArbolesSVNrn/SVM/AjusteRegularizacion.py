# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:59:17 2024

@author: joseangelperez
"""

import mglearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


import numpy as np

def make_forge():
    # a carefully hand-designed dataset lol
    X, y = make_blobs(centers=2, random_state=4, n_samples=300)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=bool)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y

X, y = make_forge()


# Hacer Split de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

# entrenamiento de Datos

fix, axes = plt.subplots(1, 2, figsize=(10, 3))
global_C = 100 # Parámetro de regularización
for model, ax in zip([LinearSVC(C=global_C), LogisticRegression(C=global_C)], axes):
    clf = model.fit(X_train, y_train)
    # mglearn.plots.plot_2d_separator(clf, X_train, fill=False, eps=.5, ax=ax, alpha=0.7)
    
    # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    
    
    # En la leyenda superior, mostrar modelo que toque en cada iteración
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("Primera característica")
    ax.set_ylabel("Segunda característica")
    
    ax.set_xticks(range(7, 14))
    ax.set_yticks(range(-1, 6))

    ax.legend()
    
    y_res=clf.predict(X_test)
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_res, ax=ax)
    """
    print(clf.predict([[9, 0.5]]))
    
    for i in y_res:
        if y_res[i] == 0:
            y_res[i]=2
        elif y_res[i] == 1:
            y_res[i]=3
        else:
            pass
    """
    

# comprobacion de resultado con Test
"""
for model, ax in zip([LinearSVC(C=global_C), LogisticRegression(C=global_C)], axes):
    clf = model.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(clf, X_train, fill=False, eps=.5, ax=ax, alpha=0.7)
    
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    
    
    # En la leyenda superior, mostrar modelo que toque en cada iteración
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("Primera característica")
    ax.set_ylabel("Segunda característica")
    
    ax.set_xticks(range(7, 14))
    ax.set_yticks(range(-1, 6))

    ax.legend()
    
    print(clf.predict([[9, 0.5]]))
    y_res=clf.predict(X_test)
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_res, ax=ax)
    
"""
    