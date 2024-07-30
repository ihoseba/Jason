# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:08:30 2024

@author: joseangelperez

Emplea la función make_biclusters para generar un dataset con 900 elementos
(30 x 30) y 4 biclústeres con una desviación estándar del ruido gaussiano de 2.
A continuación, baraja este dataset 
y procede a calcular sus biclústeres.
Determina el consensus_score (recuerda que cuando generas el dataset el 
sistema te devuelve las filas y columnas en las que están los biclústeres). 
Obtén las representaciones gráficas del dataset original, del barajado y 
del correspondiente a la solución final. 
"""

# Emplea la función make_biclusters para generar un dataset con 900 elementos
# (30 x 30) y 4 biclústeres con una desviación estándar del ruido gaussiano de 2.
from sklearn.datasets import make_biclusters
num_clusters=4
datos, filas, columnas = make_biclusters(shape=(900, 900), n_clusters=num_clusters,
                                         noise=2, shuffle=False, random_state=0)

# Mostrar
from matplotlib import pyplot as plt
plt.matshow(datos)

# A continuación, baraja este dataset 
import numpy as np
#Empleamos una variable auxiliar aleatoria, rng
rng = np.random.RandomState(0)
#Obtenemos permutaciones de las filas y las columnas
row_idx = rng.permutation(datos.shape[0])
col_idx = rng.permutation(datos.shape[1])
#Reconstruimos la matriz con la nueva distribución de filas y columnas
datos_desordenados = datos[row_idx][:, col_idx]
plt.matshow(datos_desordenados, cmap=plt.cm.Blues)

# y procede a calcular sus biclústeres.
from sklearn.cluster import SpectralCoclustering
model = SpectralCoclustering(n_clusters=num_clusters, random_state=0)
model.fit(datos_desordenados)
fit_data = datos_desordenados[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]
plt.matshow(fit_data, cmap=plt.cm.Reds)

# Determina el consensus_score (recuerda que cuando generas el dataset el 
# sistema te devuelve las filas y columnas en las que están los biclústeres). 

from sklearn.metrics import consensus_score

# Calcula el consensus_score utilizando la similitud de Jaccard
score = consensus_score(model.biclusters_, (datos[:, row_idx], datos[:, col_idx]))

print(f"Consensus Score: {score:.2f}")
# Obtén las representaciones gráficas del dataset original, del barajado y 
# del correspondiente a la solución final. 





