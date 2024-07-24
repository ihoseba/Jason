# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:20:48 2024

@author: MarkelP

Copilot en python Aplica el algoritmo DBSCAN al dataset moons 
Visualiza graficamente los puntos
Pon leyenda y x e y de lo que es cada cosa
 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Crear el conjunto de datos "moons"
X, _ = make_moons(n_samples=400, noise=0.05, random_state=0)

# Instanciar el modelo DBSCAN
model = DBSCAN(eps=0.2, min_samples=5)

# Entrenar el modelo y predecir los clusters
clusters = model.fit_predict(X)

# Graficar los puntos con colores según los clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBSCAN Clustering en el conjunto de datos "moons"')
plt.colorbar(label='Cluster')
plt.show()

