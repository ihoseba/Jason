# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:00 2024

@author: MarkelP
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generamos el conjunto de datos
X, y = make_blobs(random_state=20)

# Creamos la case KMeans y la instanciamos
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print("Los clústeres a los que pertenece casa instacia son:", kmeans.labels_)
print("Los clústeres a los que pertenece casa instacia son:", kmeans.cluster_centers_)

import numpy as np
import mglearn
import matplotlib.pyplot as plt

mglearn.discrete_scatter(X[:,0], X[:,1], kmeans.labels_, markers="o")

mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0],
                         kmeans.cluster_centers_[:, 1],
                         [0,1,2],
                         markers="^", markeredgewidth=4)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
        plt.scatter(centroids[:, 0], centroids[:, 1],
        marker='o', s=30, linewidths=8,
        color=circle_color, zorder=10, alpha=0.9)
        plt.scatter(centroids[:, 0], centroids[:, 1],
        marker='x', s=50, linewidths=50,
        color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
    np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
    cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k')
    plot_data(X)

    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)

    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
        
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plt.figure(figsize=(8, 4))

plot_decision_boundaries(kmeans, X)



X_nuevo = np.array([[0, 5]])

print(kmeans.predict(X_nuevo))



