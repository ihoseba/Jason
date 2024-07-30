# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:08:30 2024

@author: joseangelperez

Emplea la función make_ make_checkerboard para generar un dataset
(mismo ajuste que en el anterior ejemplo) y 4 biclústeres con una desviación 
estándar del ruido gaussiano de 10.
A continuación, baraja este dataset 
y procede a calcular sus biclústeres con SpectralBiclustering. 
Determina el consensus_score (recuerda que cuando generas el dataset el 
sistema te devuelve las filas y columnas en las que están los biclústeres).
Obtén las representaciones gráficas del dataset original, del barajado y
del correspondiente a la solución final. 

"""

# Emplea la función make_ make_checkerboard para generar un dataset
# (mismo ajuste que en el anterior ejemplo) y 4 biclústeres con una desviación 
# estándar del ruido gaussiano de 10.

from matplotlib import pyplot as plt
from sklearn.datasets import make_checkerboard

num_clusters = (4, 3)
datos, filas, columnas = make_checkerboard(shape=(30, 50), n_clusters=num_clusters,
                                           noise=10, shuffle=False, random_state=0)
# Mostrar
plt.matshow(datos, cmap=plt.cm.Greens)
plt.title("Conjunto original")

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

# y procede a calcular sus biclústeres con SpectralBiclustering. 
from sklearn.cluster import SpectralBiclustering
model = SpectralBiclustering(n_clusters=num_clusters, random_state=0)
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

