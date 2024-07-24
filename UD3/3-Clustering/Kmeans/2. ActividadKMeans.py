# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:32:52 2024

@author: JoseAngelPerez

* dataset iris e identifica los puntos a través de la longitud y anchura de cada pétalo, 
* ignorando el etiquetado de cada una de las especies, 
* y aplica el algoritmo de las K-medias. 
* El número de agrupamientos recomendable es 3, por supuesto, 
* pero comprueba qué resultados obtienes, además, si planteas 2 y 4 agrupamientos, respectivamente. 
* Representa dichos resultados gráficamente. 
* Para cada entrenamiento muestra la posición de los centroides; 
obtén también a qué clúster asigna el algoritmo el punto (2.5, 1) y
cuál es la distancia de dicho punto a los restantes centroides.

Copilot genera codigo python para todo lo siguiente en secuencia
Con dataset iris e identifica los puntos a través de la longitud y anchura 
de cada pétalo, e ignorando el etiquetado de cada de las especies, 
Aplica el algoritmo de las K-medias de sklearn. 
Representa en tres graficas para diferentes agrupamientos K-medias como 2, 3 y 4
mostrando tambien la posición de los centroides. Pon leyendas de ejes x e y 
tambien de los cluster como 0,1,2, etc

obtén también a qué clúster asigna el algoritmo el punto (2.5, 1) y
cuál es la distancia de dicho punto a los restantes centroides.

"""
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
iris = load_iris()

from sklearn.cluster import KMeans

# Crear una instancia de KMeans con 3 agrupamientos
kmeans = KMeans(n_clusters=3)

# Ajustar el modelo a los datos (ignorando las etiquetas de especies)
kmeans.fit(iris.data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

import matplotlib.pyplot as plt

# Crear una figura con 3 subgráficas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Graficar los puntos y los centroides
for i in range(3):
    axs[i].scatter(iris.data[:, 0], iris.data[:, 1], c=labels, cmap='viridis')
    axs[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
    axs[i].set_title(f'{i+2} agrupamientos')

plt.show()
