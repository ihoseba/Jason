# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:51:27 2024

@author: JoseAngelPerez
Actividad práctica: óptimo de agrupaciones según el algoritmo de las K-
Medias
Vuelve a considerar el dataset iris, identificando los puntos a través de la longitud y anchura de
cada pétalo e ignorando el etiquetado de cada una de las especies. Imagina que quieres
determinar el óptimo de agrupaciones según el algoritmo de las K-medias. Representa la gráfica de la inercia y del índice de siluetas para un máximo de 8 agrupamientos. Obtén también el
diagrama de silueta considerando los resultados previos. ¿Qué número de clústeres
propondrías?

Copilot, con el dataset iris, y
trabajando unicamente con los puntos de la longitud y anchura de cada pétalo
e ignorando el etiquetado de cada una de las especies.
Agrupa según el algoritmo de las K-medias barriendo casos desde 2 hasta 8 grupos.
Representa la gráfica de la inercia y del índice de siluetas para cada uno de los casos.
Obtén también el diagrama de silueta considerando los resultados previos. 

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data[:, 2:4]  # Usar solo longitud y anchura de pétalos

# Iterar sobre diferentes agrupamientos
inertia_values = []
silhouette_scores = []
silhouette_samples_list = []
for n_clusters in range(2, 9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    silhouette_samples_list.append(silhouette_samples(X, kmeans.labels_))

# Graficar la inercia
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(2, 9), inertia_values, marker='o')
plt.xlabel('Número de grupos (K)')
plt.ylabel('Inercia')
plt.title('Gráfica de Inercia')

# Graficar el índice de siluetas
plt.subplot(1, 2, 2)
plt.plot(range(2, 9), silhouette_scores, marker='o')
plt.xlabel('Número de grupos (K)')
plt.ylabel('Índice de Siluetas')
plt.title('Gráfica de Índice de Siluetas')

plt.tight_layout()
plt.show()

# Graficar el diagrama de silueta para K=3 (por ejemplo)
plt.figure(figsize=(6, 4))
silhouette_avg = silhouette_scores[1]  # Índice de siluetas para K=3
for i in range(3):
    silhouettes = silhouette_samples_list[1][kmeans.labels_ == i]
    silhouettes.sort()
    y_lower = i * len(silhouettes)
    y_upper = y_lower + len(silhouettes)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, silhouettes, alpha=0.7)
    plt.text(-0.05, (y_lower + y_upper) / 2, f'Grupo {i}', fontsize=10, va='center')
plt.axvline(x=silhouette_avg, color='red', linestyle='--')
plt.xlabel('Coeficiente de Silueta')
plt.ylabel('Muestra')
plt.title('Diagrama de Silueta para K=3')
plt.show()

# Graficar el diagrama de silueta para K=4 (por ejemplo)
la_silueta=4
plt.figure(figsize=(6, 4))
silhouette_avg = silhouette_scores[la_silueta-2]  # Índice de siluetas para K=4
for i in range(la_silueta):
    silhouettes = silhouette_samples_list[la_silueta-2][kmeans.labels_ == i]
    silhouettes.sort()
    y_lower = i * len(silhouettes)
    y_upper = y_lower + len(silhouettes)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, silhouettes, alpha=0.7)
    plt.text(-0.05, (y_lower + y_upper) / 2, f'Grupo {i}', fontsize=10, va='center')
plt.axvline(x=silhouette_avg, color='red', linestyle='--')
plt.xlabel('Coeficiente de Silueta')
plt.ylabel('Muestra')
plt.title(f'Diagrama de Silueta para K={la_silueta}')
plt.show()


