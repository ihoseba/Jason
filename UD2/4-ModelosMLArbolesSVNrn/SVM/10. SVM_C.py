# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:54:08 2024

@author: MarkelP
"""

import mglearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

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


fix, axes = plt.subplots(1, 2, figsize=(10, 3))
global_C = 1000 # Parámetro de regularización
for model, ax in zip([LinearSVC(C=global_C), LogisticRegression(C=global_C)], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=.5, ax=ax, alpha=0.7)
    
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    
    
    # En la leyenda superior, mostrar modelo que toque en cada iteración
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("Primera característica")
    ax.set_ylabel("Segunda característica")
    
    ax.set_xticks(range(7, 14))
    ax.set_yticks(range(-1, 6))

    ax.legend()
    
    print(clf.predict([[9, 0.5]]))
    
    
    
    