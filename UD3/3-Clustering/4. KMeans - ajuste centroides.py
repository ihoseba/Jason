# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:00 2024

@author: MarkelP
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

# Generamos el conjunto de datos
X, y = make_blobs(random_state=20)

centroides_inic = np.array([[-10,4], [2,8], [7,6]])

# Creamos la case KMeans y la instanciamos
kmeans = KMeans(n_clusters=3, init=centroides_inic, n_init=1)
kmeans.fit(X)

print(kmeans.inertia_)

print("Los clústeres a los que pertenece casa instacia son:", kmeans.labels_)
print("Los clústeres a los que pertenece casa instacia son:", kmeans.cluster_centers_)

import numpy as np
import mglearn
import matplotlib.pyplot as plt

y_nuevo = kmeans.labels_
mglearn.discrete_scatter(X[:,0], X[:,1], y_nuevo, markers="o")

mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0],
                         kmeans.cluster_centers_[:, 1],
                         [0,1,2],
                         markers="^", markeredgewidth=4)



