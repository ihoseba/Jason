# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:02:13 2024

@author: MarkelP
"""


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


import numpy as np
import mglearn
import matplotlib.pyplot as plt


# Generamos el conjunto de datos
centros_blob = np.array([[1.5,2.4],[0.5,2.3],[-0.5,2],[-1,3],[-1.5,2.6]])
blob_std = np.array([0.3, 0.25, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=800, centers=centros_blob, cluster_std=blob_std,random_state=20)

n_clusters = 5
# Creamos la case KMeans y la instanciamos
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)


labels_ = kmeans.labels_
mglearn.discrete_scatter(X[:,0], X[:,1], labels_, markers="o")

cl_ceneters = kmeans.cluster_centers_

mglearn.discrete_scatter(cl_ceneters[:, 0],
                         cl_ceneters[:, 1],
                         range(n_clusters),
                         markers="^", markeredgewidth=4)