# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:05:01 2024

@author: MarkelP
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

datos, filas, columnas = make_biclusters(shape=(30, 30), n_clusters=4, noise=2, shuffle=False, random_state=0)
plt.matshow(datos, cmap=plt.cm.Greens)
plt.title("Conjunto original")


#Barajamos el dataset
rng = np.random.RandomState(0)
row_idx = rng.permutation(datos.shape[0])
col_idx = rng.permutation(datos.shape[1])


#Reconstruimos la matriz con la nueva distribución de filas y columnas
datos = datos[row_idx][:, col_idx]
plt.matshow(datos, cmap=plt.cm.Blues)
plt.title("Conjunto barajado")


#Aplicamos el algoritmo
model = SpectralCoclustering(n_clusters=4, random_state=0)
model.fit(datos)
fit_data = datos[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]
plt.matshow(fit_data, cmap=plt.cm.Reds)
plt.title("Solución final")


#Cálculo de la puntuación de consenso
score = consensus_score(model.biclusters_, (filas[:, row_idx], columnas[:, col_idx]))
print("consensus score: {:.3f}".format(score))