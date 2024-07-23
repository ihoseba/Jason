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
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data  # Características (longitud y anchura de pétalos)

# Aplicar el algoritmo K-medias con diferentes agrupamientos
for n_clusters in [2, 3, 4]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Crear una figura con subgráficas
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        ax.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')

    # Graficar los centroides
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroides')

    # Configurar leyendas y etiquetas de ejes
    ax.set_title(f'K-medias con {n_clusters} agrupamientos')
    ax.set_xlabel('Longitud de pétalo')
    ax.set_ylabel('Anchura de pétalo')
    ax.legend()

    plt.show()
