# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:54:08 2024

@author: MarkelP
"""

import mglearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

#X_, X_2 = X[:,0], X[:,1]
#mglearn.discrete_scatter(X_, X_2, y)

fix, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
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
    
    
    
    