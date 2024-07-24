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

kmeans_por_k = [
    KMeans(n_clusters=k, random_state=20).fit(X) for k in range(1, 11)
]

'''
Lo que hace es calcular la inercia de cada modelo, que es la distancia cuadrática
media de cada instancia a su centroide más cercano, es decir, al que define el clúster
en el que se encuentra clasificado. La mejor solución es la que posee una inercia más baja.
Para conocer el valor de la inercia de un modelo solo tienes que llamar a la variable inertia_:
'''

inertiras = [m.inertia_ for m in kmeans_por_k]


plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 11), inertiras, "bo-")
plt.xlabel('Valores de K')
plt.ylabel('Inercias')